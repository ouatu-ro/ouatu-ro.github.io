// @ts-check
import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

// https://astro.build/config
export default defineConfig({
  site: "https://ouatu.ro/",
  integrations: [mdx(), sitemap()],
  base: "/",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      [
        rehypeKatex,
        {
          // Katex options
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
});
