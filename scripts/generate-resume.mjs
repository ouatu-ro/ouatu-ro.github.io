#!/usr/bin/env node

/**
 * Generate the LaTeX source and PDF resume from
 * `src/content/resume/resume.json`.
 *
 * - Renders `src/content/resume/resume.tex`
 * - Runs `latexmk -pdf` to emit `src/content/resume/resume.pdf`
 * - Copies the PDF to `public/resume.pdf`
 *
 * After the PDF is built you can optionally compare it with
 * `resume-target.pdf` via `pdftotext` to ensure parity.
 */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const RESUME_DIR = path.join(ROOT, "src/content/resume");

const RESUME_JSON = path.join(RESUME_DIR, "resume.json");
const RESUME_TEX = path.join(RESUME_DIR, "resume.tex");
const RESUME_PDF = path.join(RESUME_DIR, "resume.pdf");
const TARGET_PDF = path.join(RESUME_DIR, "resume-target.pdf");
const PUBLIC_PDF = path.join(ROOT, "public/resume.pdf");

async function main() {
  const resume = await readJson(RESUME_JSON);

  await fs.writeFile(RESUME_TEX, renderLatex(resume), "utf8");
  await buildPdf();
  await copyPdfToPublic();

  await maybeCompareWithTarget();
}

async function readJson(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  return JSON.parse(raw);
}

function renderLatex(resume) {
  const { basics } = resume;
  const header = `\\documentclass[11pt,a4paper]{article}

\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{lmodern}
\\usepackage{microtype}
\\usepackage{ragged2e}
\\usepackage[a4paper,margin=1.8cm]{geometry}
\\usepackage{xcolor}
\\definecolor{accent}{HTML}{0F766E}
\\definecolor{ink}{HTML}{111827}
\\definecolor{muted}{HTML}{6B7280}
\\definecolor{rulec}{HTML}{E5E7EB}
\\usepackage[hidelinks]{hyperref}
\\hypersetup{colorlinks=true,urlcolor=accent,linkcolor=accent}
\\usepackage{enumitem}
\\setlist[itemize]{leftmargin=1.2em,itemsep=0.25em,topsep=0.2em}
\\usepackage{tabularx}
\\usepackage{array}
\\usepackage{booktabs}
\\RaggedRight
\\setlength{\\parindent}{0pt}
\\setlength{\\parskip}{0.2em}
\\newcommand{\\sectionhead}[1]{%
  \\vspace{1.0em}%
  {\\large\\bfseries\\textcolor{ink}{\\MakeUppercase{#1}}}\\par
  {\\color{rulec}\\rule{\\linewidth}{0.8pt}}\\vspace{0.4em}
}
\\newcommand{\\tworow}[2]{%
  \\noindent\\begin{tabularx}{\\linewidth}{@{}X>{\\raggedleft\\arraybackslash}p{.30\\linewidth}@{}}
    #1 & {\\color{muted}#2}
  \\end{tabularx}\\vspace{-0.1em}
}
\\newcommand{\\chipline}[1]{\\textcolor{muted}{#1}}

\\begin{document}

\\begin{center}
  {\\LARGE\\bfseries \\textcolor{ink}{${tex(basics.name)}}}\\par
  \\vspace{0.25em}
  {\\large\\color{accent} ${tex(basics.headline)}}\\par
  \\vspace{0.8em}
  \\small
  ${renderContactLine(resume)}
\\end{center}
`;

  const summary = `\\sectionhead{Summary}
${formatSummary(resume.summary)}
`;

  const selectedResults = renderSelectedResults(resume.selectedResults);
  const experience = renderExperience(resume.experience);
  const consulting = renderConsulting(resume.consultingServices);
  const education = renderEducation(resume.education);
  const awards = renderAwards(resume.awards);

  return `${latexComment()}
${header}
% =====================================================================
% Summary
% =====================================================================
${summary}
% =====================================================================
% Selected Results
% =====================================================================
${selectedResults}
% =====================================================================
% Experience
% =====================================================================
${experience}
% =====================================================================
% Consulting Services
% =====================================================================
${consulting}
% =====================================================================
% Education & Awards
% =====================================================================

${education}

${awards}

\\vspace{0.5em}

\\end{document}
`;
}

function latexComment() {
  return `% =====================================================================
% Bogdan Ouatu — Resume (LaTeX, generated)
% Built from resume.json via scripts/generate-resume.mjs
% =====================================================================
`;
}

function renderContactLine(resume) {
  const segments = collectContactEntries(resume.basics).map((entry) => {
    if (entry.kind === "text") return tex(entry.label);
    const href = entry.kind === "mailto" ? entry.url : escapeUrl(entry.url);
    return `\\href{${href}}{${tex(entry.label)}}`;
  });

  return segments.join("\n  \\,|\\, ");
}

function renderSelectedResults(section) {
  if (!section?.items?.length) return "";
  const items = section.items
    .map(
      (item) =>
        `  \\item \\textbf{${tex(item.title)}:} ${formatInline(item.description)}`,
    )
    .join("\n");

  return `\\sectionhead{${tex(section.name)}}

\\begin{itemize}
${items}
\\end{itemize}
`;
}

function renderExperience(section) {
  if (!section?.items?.length) return "";

  const items = section.items
    .map((job) => {
      const header = `\\tworow{{\\bfseries ${tex(job.position)} --- ${renderCompany(job)}}}{${formatDate(
        job.date,
      )}}`;
      const bullets = (job.highlights ?? [])
        .map((line) => `  \\item ${formatBullet(line)}`)
        .join("\n");
      return `
${header}
\\begin{itemize}
${bullets}
\\end{itemize}
`;
    })
    .join("\n");

  return `\\sectionhead{${tex(section.name)}}
${items}`.trimEnd();
}

function renderCompany(job) {
  if (job.url) {
    return `\\href{${escapeUrl(job.url)}}{${tex(job.company)}}`;
  }
  return tex(job.company);
}

function renderConsulting(section) {
  if (!section?.items?.length) return "";

  const rows = [];
  const items = section.items;
  for (let i = 0; i < items.length; i += 2) {
    const left = items[i];
    const right = items[i + 1];
    rows.push(
      `  ${renderConsultingCell(left)}\n  &\n  ${right ? renderConsultingCell(right) : ""}\n  \\\\`,
    );
    if (i + 2 < items.length) {
      rows.push("  \\addlinespace[0.6em]");
    }
  }

  return `\\sectionhead{${tex(section.name)}}

\\begin{tabularx}{\\linewidth}{@{}X X@{}}
${rows.join("\n")}
\\end{tabularx}
`;
}

function renderConsultingCell(service) {
  if (!service) return "";
  const keywords = service.keywords ?? [];
  return `\\textbf{${tex(service.category)}} \\par
  \\chipline{${tex(keywords.join("; "))}}`;
}

function renderEducation(section) {
  if (!section?.items?.length) return "";

  const items = section.items
    .map(
      (item) =>
        `  \\item \\textbf{${tex(item.institution)}} --- ${tex(item.degree)} \\hfill \\textcolor{muted}{${tex(
          item.year,
        )}}`,
    )
    .join("\n");

  return `\\sectionhead{${tex(section.name)}}
\\begin{itemize}
${items}
\\end{itemize}
`;
}

function renderAwards(section) {
  if (!section?.items?.length) return "";

  const items = section.items
    .map(
      (item) =>
        `  \\item \\textbf{${tex(item.title)}} --- ${tex(item.awarder)} \\hfill \\textcolor{muted}{${tex(
          item.year,
        )}}`,
    )
    .join("\n");

  return `\\sectionhead{${tex(section.name)}}
\\begin{itemize}
${items}
\\end{itemize}
`;
}

function formatSummary(text) {
  let safe = escapeLatex(text);
  safe = enDashNumbers(safe);
  safe = safe.replace(/3--10x faster/gi, "\\textbf{3--10x faster}");
  safe = safe.replace(/30\\% more cost-efficient/gi, "\\textbf{30\\% more cost-efficient}");
  safe = safe.replace(/60k\+ users/gi, "\\textbf{60k+ users}");
  return safe;
}

function formatInline(text) {
  let safe = escapeLatex(text);
  safe = enDashNumbers(safe);
  return safe;
}

function formatBullet(text) {
  let safe = escapeLatex(text);
  safe = enDashNumbers(safe);

  const colonIndex = safe.indexOf(":");
  if (colonIndex !== -1) {
    const before = safe.slice(0, colonIndex + 1);
    const after = safe.slice(colonIndex + 1);
    safe = `\\textbf{${before}}${after}`;
  }

  return safe;
}

function formatDate(text) {
  return escapeLatex(text).replace(/ - /g, " -- ");
}

function enDashNumbers(value) {
  return value.replace(/(\d)\-(\d)/g, "$1--$2");
}

function tex(value = "") {
  return escapeLatex(value);
}

function escapeLatex(value = "") {
  return String(value)
    .replace(/\\/g, "\\textbackslash{}")
    .replace(/([#$%&_{}])/g, "\\$1")
    .replace(/~/g, "\\textasciitilde{}")
    .replace(/\^/g, "\\textasciicircum{}");
}

function profileLabel(profile) {
  if (profile.urlLabel) return profile.urlLabel;
  const network = profile.network?.toLowerCase();
  try {
    const parsed = new URL(profile.url);
    const host = parsed.hostname.replace(/^www\./, "");
    if (network === "github") {
      const path = parsed.pathname.replace(/\/$/, "");
      return `${host}${path}`;
    }
    return host;
  } catch {
    return profile.network || profile.url;
  }
}

function collectContactEntries(basics) {
  const profiles = basics.profiles ?? [];
  const website = profiles.find(
    (p) => (p.network ?? "").toLowerCase() === "website",
  );
  const github = profiles.find((p) => (p.network ?? "").toLowerCase() === "github");

  const entries = [];

  if (website) {
    entries.push({
      kind: "link",
      url: website.url,
      label: profileLabel(website),
    });
  }

  if (basics.email) {
    entries.push({
      kind: "mailto",
      url: `mailto:${basics.email}`,
      label: basics.email,
    });
  }

  if (basics.location) {
    entries.push({ kind: "text", label: basics.location });
  }

  for (const profile of profiles) {
    if (profile === website || profile === github) continue;
    entries.push({
      kind: "link",
      url: profile.url,
      label: profileLabel(profile),
    });
  }

  if (github) {
    entries.push({
      kind: "link",
      url: github.url,
      label: profileLabel(github),
    });
  }

  return entries;
}

async function buildPdf() {
  await new Promise((resolve, reject) => {
    const proc = spawn("latexmk", ["-pdf", "resume.tex"], {
      cwd: RESUME_DIR,
      stdio: "inherit",
    });
    proc.on("error", (error) => reject(new Error(`latexmk execution failed: ${error.message}`)));
    proc.on("exit", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`latexmk exited with code ${code}`));
      }
    });
  });

  const auxArtifacts = [
    "resume.aux",
    "resume.fdb_latexmk",
    "resume.fls",
    "resume.log",
    "resume.out",
    "resume.synctex.gz",
  ];

  for (const name of auxArtifacts) {
    const filePath = path.join(RESUME_DIR, name);
    try {
      await fs.unlink(filePath);
    } catch {}
  }
}

async function copyPdfToPublic() {
  await fs.mkdir(path.dirname(PUBLIC_PDF), { recursive: true });
  await fs.copyFile(RESUME_PDF, PUBLIC_PDF);
}

async function maybeCompareWithTarget() {
  try {
    await fs.access(TARGET_PDF);
  } catch {
    return;
  }

  console.log("Tip: run `pdftotext` on resume-target.pdf and resume.pdf to compare content:");
  console.log(
    `  pdftotext ${path.relative(process.cwd(), TARGET_PDF)} target.txt && pdftotext ${path.relative(
      process.cwd(),
      RESUME_PDF,
    )} current.txt && diff -u target.txt current.txt`,
  );
}

function escapeUrl(value = "") {
  return String(value).replace(/\s/g, "%20");
}

await main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
