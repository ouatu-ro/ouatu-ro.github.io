// @ts-check
import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import viteCompression from "vite-plugin-compression"; // 🔥 add this
import fs from "fs";
import path from "path";

// Load custom sitemap URLs
let projectUrls = [];
try {
  const projectUrlsPath = path.join(
    process.cwd(),
    "public",
    "project-urls.json"
  );
  if (fs.existsSync(projectUrlsPath)) {
    const data = JSON.parse(fs.readFileSync(projectUrlsPath, "utf8"));
    projectUrls = data.urls || [];
    console.log(`Loaded ${projectUrls.length} project URLs for sitemap`);
  }
} catch (error) {
  console.warn("Could not load project URLs for sitemap:", error);
}

// https://astro.build/config
export default defineConfig({
  site: "https://ouatu.ro/",
  integrations: [
    mdx(),
    sitemap({
      customPages: projectUrls,
      changefreq: "weekly",
      priority: 0.5,
      lastmod: new Date(),
    }),
  ],
  base: "/",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      [
        rehypeKatex,
        {
          // throwOnError: false,
          output: "html",
          displayMode: true,
        },
      ],
    ],
    shikiConfig: {
      theme: "github-dark",
    },
  },
  vite: {
    build: {
      assetsInlineLimit: 65536,
    },
    plugins: [
      viteCompression({
        algorithm: "brotliCompress", // or 'gzip'
        ext: ".br",
        filter: /\.(js|mjs|json|css|html|ico|woff2)$/,
        deleteOriginFile: false,
      }),
    ],
  },
});
