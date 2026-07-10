use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use std::path::{Path, PathBuf};
use sundial_core::problem::{ProgressEvent, Solution, SolveOptions};

mod report;

#[derive(Parser)]
#[command(
    name = "sundial",
    version,
    about = "WebGPU-native LP solver (restarted PDHG)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Engine {
    Gpu,
    Cpu,
}

#[derive(Subcommand)]
enum Cmd {
    /// Solve a single MPS file
    Solve {
        file: PathBuf,
        #[arg(long, default_value_t = 1e-4)]
        tol: f64,
        #[arg(long, value_enum, default_value_t = Engine::Gpu)]
        engine: Engine,
        #[arg(long, default_value_t = 2_000_000)]
        max_iters: u64,
        #[arg(long)]
        json: bool,
    },
    /// Solve every *.mps / *.mps.gz in a directory, write a CSV
    Bench {
        dir: PathBuf,
        #[arg(long, default_value_t = 1e-4)]
        tol: f64,
        #[arg(long, value_enum, default_value_t = Engine::Gpu)]
        engine: Engine,
        #[arg(long, default_value = "results.csv")]
        out: PathBuf,
    },
    /// Render a bench results CSV + known optima into report.md
    Report {
        csv: PathBuf,
        #[arg(long)]
        optima: Option<PathBuf>,
        #[arg(long, default_value = "report.md")]
        out: PathBuf,
    },
    /// Solve a generated optimal-transport instance (the M1 hero)
    Transport {
        #[arg(long, default_value_t = 32)]
        grid: usize,
        #[arg(long, default_value = "blobs")]
        preset: String,
        #[arg(long, default_value_t = 1e-4)]
        tol: f64,
        #[arg(long, value_enum, default_value_t = Engine::Gpu)]
        engine: Engine,
        #[arg(long, default_value_t = 500_000)]
        max_iters: u64,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Serialize)]
struct Report<'a> {
    name: &'a str,
    status: String,
    objective: f64,
    iterations: u64,
    restarts: u32,
    solve_ms: f64,
    rel_primal: f64,
    rel_dual: f64,
    rel_gap: f64,
}

fn report<'a>(name: &'a str, s: &Solution) -> Report<'a> {
    Report {
        name,
        status: format!("{:?}", s.status),
        objective: s.primal_obj,
        iterations: s.stats.iterations,
        restarts: s.stats.restarts,
        solve_ms: s.stats.solve_ms,
        rel_primal: s.stats.verified.rel_primal,
        rel_dual: s.stats.verified.rel_dual,
        rel_gap: s.stats.verified.rel_gap,
    }
}

fn csv_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

fn solve_file(
    path: &Path,
    tol: f64,
    max_iters: u64,
    engine: Engine,
    quiet: bool,
) -> Result<Solution> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let p =
        sundial_mps::parse_bytes(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    let opts = SolveOptions {
        tol,
        max_iters,
        ..Default::default()
    };
    let mut on_progress = |e: ProgressEvent| {
        if !quiet {
            eprintln!(
                "iter {:>8}  primal {:.2e}  dual {:.2e}  gap {:.2e}  {:.3} ms/iter",
                e.iter, e.rel_primal, e.rel_dual, e.rel_gap, e.ms_per_iter
            );
        }
    };
    Ok(match engine {
        Engine::Cpu => sundial_core::reference::solve(&p, &opts, &mut on_progress),
        Engine::Gpu => {
            let ctx = pollster::block_on(sundial_core::gpu::GpuContext::new())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            eprintln!(
                "GPU: {} (max binding {} MiB)",
                ctx.adapter_name, ctx.max_binding_mib
            );
            pollster::block_on(sundial_core::gpu::engine::solve_gpu(
                &ctx,
                &p,
                &opts,
                &mut on_progress,
            ))
            .map_err(|e| anyhow::anyhow!("{e}"))?
        }
    })
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Solve {
            file,
            tol,
            engine,
            max_iters,
            json,
        } => {
            let sol = solve_file(&file, tol, max_iters, engine, json)?;
            let name = file.file_stem().unwrap_or_default().to_string_lossy();
            if json {
                println!("{}", serde_json::to_string_pretty(&report(&name, &sol))?);
            } else {
                println!(
                    "{name}: {:?}  obj {:.10}  ({} iters, {} restarts, {:.0} ms, verified mu {:.2e})",
                    sol.status, sol.primal_obj, sol.stats.iterations, sol.stats.restarts,
                    sol.stats.solve_ms, sol.stats.verified.mu()
                );
            }
            match sol.status {
                sundial_core::problem::SolveStatus::Optimal => {}
                sundial_core::problem::SolveStatus::Infeasible => {
                    bail!("infeasible (Farkas ray certified in f64)")
                }
                sundial_core::problem::SolveStatus::Unbounded => {
                    bail!("unbounded (improving ray certified in f64)")
                }
                _ => bail!("not solved to tolerance"),
            }
        }
        Cmd::Bench {
            dir,
            tol,
            engine,
            out,
        } => {
            let mut rows = vec![
                "name,status,objective,iterations,solve_ms,rel_primal,rel_dual,rel_gap".to_string(),
            ];
            let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.to_string_lossy().ends_with(".mps")
                        || p.to_string_lossy().ends_with(".mps.gz")
                })
                .collect();
            files.sort();
            if files.is_empty() {
                bail!("no .mps files in {}", dir.display());
            }
            for f in files {
                let name = f
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                match solve_file(&f, tol, 2_000_000, engine, true) {
                    Ok(s) => {
                        let r = report(&name, &s);
                        println!(
                            "{name}: {} obj {:.6} ({:.0} ms)",
                            r.status, r.objective, r.solve_ms
                        );
                        rows.push(format!(
                            "{},{},{},{},{:.1},{:.3e},{:.3e},{:.3e}",
                            csv_quote(r.name),
                            r.status,
                            r.objective,
                            r.iterations,
                            r.solve_ms,
                            r.rel_primal,
                            r.rel_dual,
                            r.rel_gap
                        ));
                    }
                    Err(e) => {
                        println!("{name}: ERROR {e:#}");
                        let chain = format!("{e:#}").replace(',', ";");
                        rows.push(format!("{},Error: {chain},,,,,,", csv_quote(&name)));
                    }
                }
            }
            std::fs::write(&out, rows.join("\n") + "\n")?;
            println!("wrote {}", out.display());
        }
        Cmd::Report { csv, optima, out } => {
            let csv_text = std::fs::read_to_string(&csv)
                .with_context(|| format!("reading {}", csv.display()))?;
            let optima_text = match optima {
                Some(p) => std::fs::read_to_string(&p)
                    .with_context(|| format!("reading {}", p.display()))?,
                None => include_str!("../data/netlib_optima.csv").to_string(),
            };
            let md = report::render(&csv_text, &report::parse_optima(&optima_text));
            std::fs::write(&out, &md)?;
            println!("wrote {}", out.display());
        }
        Cmd::Transport {
            grid,
            preset,
            tol,
            engine,
            max_iters,
            json,
        } => {
            let preset: sundial_core::transport::Preset =
                preset.parse().map_err(|e: String| anyhow::anyhow!(e))?;
            let p = sundial_core::transport::problem(preset, grid);
            eprintln!(
                "transport {grid}x{grid}: {} variables, {} constraints",
                p.n_vars(),
                p.n_cons()
            );
            let opts = SolveOptions {
                tol,
                max_iters,
                ..Default::default()
            };
            let mut on_progress = |e: ProgressEvent| {
                if !json {
                    eprintln!(
                        "iter {:>8}  primal {:.2e}  dual {:.2e}  gap {:.2e}  {:.3} ms/iter",
                        e.iter, e.rel_primal, e.rel_dual, e.rel_gap, e.ms_per_iter
                    );
                }
            };
            let sol = match engine {
                Engine::Cpu => sundial_core::reference::solve_op(&p, &opts, &mut on_progress),
                Engine::Gpu => {
                    let ctx = pollster::block_on(sundial_core::gpu::GpuContext::new())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    eprintln!(
                        "GPU: {} (max binding {} MiB)",
                        ctx.adapter_name, ctx.max_binding_mib
                    );
                    let gop = sundial_core::gpu::op::TransportGpuOp::new(
                        &ctx.device,
                        grid * grid,
                        grid * grid,
                    );
                    pollster::block_on(sundial_core::gpu::engine::solve_gpu_op(
                        &ctx,
                        &p,
                        &gop,
                        &opts,
                        &mut on_progress,
                        None,
                    ))
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                }
            };
            let name = format!("transport-{preset:?}-{grid}").to_lowercase();
            if json {
                println!("{}", serde_json::to_string_pretty(&report(&name, &sol))?);
            } else {
                println!(
                    "{name}: {:?}  obj {:.10}  ({} iters, {} restarts, {:.0} ms, verified mu {:.2e})",
                    sol.status, sol.primal_obj, sol.stats.iterations, sol.stats.restarts,
                    sol.stats.solve_ms, sol.stats.verified.mu()
                );
            }
            match sol.status {
                sundial_core::problem::SolveStatus::Optimal => {}
                sundial_core::problem::SolveStatus::Infeasible => {
                    bail!("infeasible (Farkas ray certified in f64)")
                }
                sundial_core::problem::SolveStatus::Unbounded => {
                    bail!("unbounded (improving ray certified in f64)")
                }
                _ => bail!("not solved to tolerance"),
            }
        }
    }
    Ok(())
}
