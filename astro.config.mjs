// @ts-check
import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";

// https://astro.build/config
export default defineConfig({
  site: "https://ouatu.ro/",
  integrations: [mdx(), sitemap()],
  base: "/",
  markdown: {
    shikiConfig: {
      theme: "github-dark",
    },
  },
});
