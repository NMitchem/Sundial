import { defineConfig } from "vite";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  base: "./",
  server: { fs: { allow: [".."] } },
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        bench: fileURLToPath(new URL("./bench.html", import.meta.url)),
        taxi: fileURLToPath(new URL("./taxi.html", import.meta.url)),
      },
    },
  },
});
