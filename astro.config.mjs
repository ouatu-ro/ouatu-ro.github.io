import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

import fs from "fs";
import path from "path";
import matter from "gray-matter";

const projectUrlsPath = path.join(process.cwd(), "public", "project-urls.json");
let projectUrls = [];
if (fs.existsSync(projectUrlsPath)) {
  try {
    projectUrls =
      JSON.parse(fs.readFileSync(projectUrlsPath, "utf8")).urls ?? [];
    console.log(`Loaded ${projectUrls.length} project URLs for sitemap`);
  } catch (e) {
    console.warn("Could not parse project-urls.json:", e);
  }
}

function readProjectsMeta() {
  const tryFiles = [
    path.join(process.cwd(), "public", "projects-data.json"),
    // path.join(process.cwd(), "public", "manual-projects-data.json"),
  ];
  for (const f of tryFiles) {
    if (fs.existsSync(f)) {
      try {
        const raw = JSON.parse(fs.readFileSync(f, "utf8"));
        const arr = raw.projects ?? raw.manualProjects ?? [];
        return arr.map((p) => ({
          homepage: p.homepage,
          slug: p.slug || (p.name ? slugify(p.name) : undefined),
          pubDate: p.pubDate ? new Date(p.pubDate) : undefined,
          updatedDate: p.updatedDate ? new Date(p.updatedDate) : undefined,
        }));
      } catch (e) {
        console.warn("Could not parse projects metadata:", e);
      }
    }
  }
  return [];
}

function slugify(text) {
  return String(text)
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^\w\-]+/g, "")
    .replace(/\-\-+/g, "-")
    .trim();
}

const projectsMeta = readProjectsMeta();

// --- blog slug -> date
function buildBlogDateMap() {
  const dir = path.join(process.cwd(), "src", "content", "blog");
  const map = new Map();
  if (!fs.existsSync(dir)) return map;
  const walk = (d) => {
    for (const f of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, f.name);
      if (f.isDirectory()) walk(p);
      else if (/\.(md|mdx)$/.test(f.name)) {
        const slug = f.name.replace(/\.(md|mdx)$/, "");
        try {
          const src = fs.readFileSync(p, "utf8");
          const fm = matter(src).data || {};
          const updated = fm.updatedDate ? new Date(fm.updatedDate) : undefined;
          const pub = fm.pubDate ? new Date(fm.pubDate) : undefined;
          map.set(slug, updated || pub);
        } catch {}
      }
    }
  };
  walk(dir);
  return map;
}
const blogDates = buildBlogDateMap();

// --- file mtime fallback
function mtimeForRoute(urlPath) {
  const candidates = [
    urlPath.endsWith("/") ? urlPath.slice(0, -1) : urlPath,
    urlPath.endsWith("/") ? urlPath + "index" : urlPath + "/index",
  ];
  const roots = ["src/pages", "src/content/blog"];
  const exts = [".astro", ".mdx", ".md"];
  for (const root of roots) {
    for (const c of candidates) {
      for (const ext of exts) {
        const p = path.join(process.cwd(), root, c + ext);
        if (fs.existsSync(p)) {
          try {
            return fs.statSync(p).mtime;
          } catch {}
        }
      }
    }
  }
  return undefined;
}

function buildHomepageMap(meta) {
  const map = new Map();
  for (const p of meta) {
    if (!p.homepage) continue;
    try {
      const u = new URL(p.homepage, "https://ouatu.ro");
      const pathname = u.pathname.endsWith("/") ? u.pathname : u.pathname + "/";
      if (pathname === "/") continue; // <- FIX: skip root
      map.set(pathname, p);
    } catch {}
  }
  return map;
}
const homepageMap = buildHomepageMap(projectsMeta);

function maxDate(...ds) {
  const ts = ds.filter(Boolean).map((d) => d.getTime());
  return ts.length ? new Date(Math.max(...ts)) : undefined;
}
const latestBlogDate = [...blogDates.values()].reduce(
  (a, b) => (a && a > b ? a : b),
  undefined
);
const latestProjectDate = projectsMeta
  .map((p) => p.updatedDate || p.pubDate)
  .reduce((a, b) => (a && a > b ? a : b), undefined);

export default defineConfig({
  site: "https://ouatu.ro/",
  base: "/",
  // trailingSlash: "always", // optional but consistent with sitemap normalization
  integrations: [
    mdx(),
    sitemap({
      customPages: projectUrls,
      changefreq: "weekly",
      priority: 0.5,
      serialize(item) {
        const url = new URL(item.url, "https://ouatu.ro/");
        // FIX: normalize incoming path once
        const pathname = url.pathname.endsWith("/")
          ? url.pathname
          : url.pathname + "/";

        // Homepage -> max(home mtime, latest blog, latest project)
        if (pathname === "/") {
          const homeMtime = mtimeForRoute("/");
          const d = maxDate(homeMtime, latestBlogDate, latestProjectDate);
          if (d) return { ...item, lastmod: d };
          return item;
        }

        // Project standalone homepages (e.g., /pong/)
        const homeMeta = homepageMap.get(pathname);
        if (homeMeta) {
          const d = homeMeta.updatedDate || homeMeta.pubDate;
          if (d) return { ...item, lastmod: d };
        }

        // blog/<slug>/
        const blogMatch = pathname.match(/^\/blog\/([^/]+)\/$/);
        if (blogMatch) {
          const d = blogDates.get(blogMatch[1]);
          if (d) return { ...item, lastmod: d };
        }

        // /project/<slug>/
        const projectMatch = pathname.match(/^\/project\/([^/]+)\/$/);
        if (projectMatch) {
          const slug = projectMatch[1];
          const meta = projectsMeta.find(
            (p) =>
              p.slug === slug ||
              (p.homepage && p.homepage.includes(`/project/${slug}/`))
          );
          const d = meta?.updatedDate || meta?.pubDate;
          if (d) return { ...item, lastmod: d };
        }

        // fallback: source mtime
        const mt = mtimeForRoute(pathname);
        if (mt) return { ...item, lastmod: mt };

        return item;
      },
    }),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { output: "html", displayMode: true }]],
    shikiConfig: { theme: "github-dark" },
  },
  vite: {
    build: { assetsInlineLimit: 65536 },
    plugins: [],
  },
});
