#!/usr/bin/env node
/* eslint-disable no-undef */
const fs = require("fs");
const path = require("path");

// ---- utils -----------------------------------------------------------------
function slugify(text) {
  return String(text)
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^\w\-]+/g, "")
    .replace(/\-\-+/g, "-")
    .trim();
}

function toISO(d) {
  try {
    return new Date(d).toISOString();
  } catch {
    return new Date().toISOString();
  }
}

function ensureTrailingSlash(u) {
  return u.endsWith("/") ? u : u + "/";
}

function isSameHost(url, host) {
  try {
    return new URL(url).host === host;
  } catch {
    return false;
  }
}

function dedupeProjects(arr) {
  const seen = new Map(); // key by slug first, then homepage
  const out = [];
  for (const p of arr) {
    const key = p.slug || p.homepage || p.name;
    if (!key) continue;
    if (seen.has(key)) continue;
    seen.set(key, true);
    out.push(p);
  }
  // secondary pass: also avoid dup by homepage
  const seenHome = new Set();
  return out.filter((p) => {
    const h = p.homepage || "";
    if (!h) return true;
    if (seenHome.has(h)) return false;
    seenHome.add(h);
    return true;
  });
}

// ---- read manual projects (do NOT overwrite) --------------------------------
function readManual() {
  const p = path.join(process.cwd(), "public", "manual-projects-data.json");
  if (!fs.existsSync(p)) return [];
  try {
    const data = JSON.parse(fs.readFileSync(p, "utf8"));
    const list = Array.isArray(data?.manualProjects) ? data.manualProjects : [];
    // normalize
    return list.map((pr) => {
      const slug = pr.slug || slugify(pr.name || "");
      const homepage = pr.homepage
        ? pr.homepage.replace(/^http:\/\//i, "https://")
        : "";
      return {
        name: pr.name || slug,
        homepage,
        description: pr.description || "",
        githubUrl: pr.githubUrl || null,
        slug,
        pubDate: pr.pubDate
          ? toISO(pr.pubDate)
          : toISO(Date.now() - 2 * 24 * 3600 * 1000),
        updatedDate: pr.updatedDate ? toISO(pr.updatedDate) : undefined,
      };
    });
  } catch (e) {
    console.warn("⚠️ Could not parse manual-projects-data.json:", e.message);
    return [];
  }
}

// ---- GitHub fetch with pagination -------------------------------------------
async function ghFetch(url) {
  const headers = { "User-Agent": "ouatu-ro-build" };
  if (process.env.GITHUB_TOKEN)
    headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  const r = await fetch(url, { headers });
  if (!r.ok)
    throw new Error(`GitHub ${r.status} ${r.statusText}: ${await r.text()}`);
  return { json: await r.json(), headers: r.headers };
}

async function fetchAllRepos(user) {
  const base = `https://api.github.com/users/${user}/repos?per_page=100&type=owner&sort=updated`;
  let url = base;
  const all = [];
  for (let page = 1; page <= 10; page++) {
    // safety cap
    const { json, headers } = await ghFetch(url + `&page=${page}`);
    all.push(...json);
    // stop if less than per_page returned
    if (!Array.isArray(json) || json.length < 100) break;
    // try Link header if present
    const link = headers.get("link");
    if (!link || !link.includes('rel="next"')) break;
  }
  return all;
}

// ---- main -------------------------------------------------------------------
(async () => {
  const githubUsername = process.argv[2] || "ouatu-ro";
  console.log(`🔎 Fetching GitHub repos for: ${githubUsername}`);

  const manual = readManual();
  console.log(`📄 Manual projects: ${manual.length}`);

  // fetch repos
  const repos = await fetchAllRepos(githubUsername);
  console.log(`📦 Repos fetched: ${repos.length}`);

  // filter to repos with a homepage, not forks/archived
  const githubProjects = repos
    .filter(
      (r) => r && !r.fork && !r.archived && r.homepage && r.homepage.trim(),
    )
    .map((r) => {
      const homepage = r.homepage.replace(/^http:\/\//i, "https://");
      return {
        name: r.name,
        homepage,
        description: r.description || "",
        githubUrl: r.html_url,
        slug: slugify(r.name),
        pubDate: toISO(r.created_at || Date.now()),
        updatedDate: toISO(r.updated_at || Date.now()),
      };
    });

  console.log(`✅ GitHub projects w/ homepage: ${githubProjects.length}`);

  // manual wins on conflicts
  const bySlug = new Map(manual.map((p) => [p.slug, p]));
  const merged = [...manual];

  for (const gp of githubProjects) {
    const m = bySlug.get(gp.slug);
    if (m) continue; // keep manual
    merged.push(gp);
  }

  // final dedupe pass (slug/homepage)
  const projects = dedupeProjects(merged);

  // write projects-data.json
  const publicDir = path.join(process.cwd(), "public");
  fs.mkdirSync(publicDir, { recursive: true });

  const projectsDataPath = path.join(publicDir, "projects-data.json");
  fs.writeFileSync(projectsDataPath, JSON.stringify({ projects }, null, 2));

  // build project-urls.json (canonical + same-host homepage)
  const site = new URL("https://ouatu.ro");
  const urls = [];
  for (const p of projects) {
    if (!p.slug) continue;
    // canonical project page
    urls.push(
      ensureTrailingSlash(new URL(`/project/${p.slug}/`, site).toString()),
    );
    // if homepage is on same host and not root, include it too
    if (p.homepage && isSameHost(p.homepage, site.host)) {
      const hp = ensureTrailingSlash(p.homepage);
      if (hp !== site.origin + "/") urls.push(hp);
    }
  }
  // dedupe urls
  const dedupedUrls = Array.from(new Set(urls));
  const projectUrlsPath = path.join(publicDir, "project-urls.json");
  fs.writeFileSync(
    projectUrlsPath,
    JSON.stringify({ urls: dedupedUrls }, null, 2),
  );

  // post-write assertions
  if (!fs.existsSync(projectsDataPath)) {
    console.error("❌ projects-data.json not created");
    process.exit(1);
  }
  const count = (
    JSON.parse(fs.readFileSync(projectsDataPath, "utf8")).projects || []
  ).length;
  console.log(`🎯 Wrote ${count} projects to public/projects-data.json`);
  console.log(
    `🔗 Wrote ${dedupedUrls.length} URLs to public/project-urls.json`,
  );

  // preview list
  for (const p of projects) {
    console.log(`- ${p.name} (${p.slug}) :: ${p.homepage}`);
  }
})().catch((err) => {
  console.error("❌ generator failed:", err);
  process.exit(1);
});
