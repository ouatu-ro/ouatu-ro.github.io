import fs from "node:fs";
import path from "node:path";
import { getCollection } from "astro:content";
import { projects as projectsConfig } from "../config/site";
import { slugify } from "./slugify";
import type { PreviewArtworkType } from "../components/artwork-manager";

export interface ProjectRecord {
  name: string;
  homepage: string;
  description: string;
  githubUrl: string | null;
  slug: string;
  pubDate?: string;
  updatedDate?: string;
  previewAsset: string | null;
  shouldShowPreview: boolean;
  previewType: PreviewArtworkType;
  isHighlight: boolean;
  highlightOrder: number;
  hasExtraContent: boolean;
}

const PROJECTS_DATA_PATH = path.join(
  process.cwd(),
  "public",
  "projects-data.json",
);

function readProjectsFile(): any[] {
  try {
    if (!fs.existsSync(PROJECTS_DATA_PATH)) {
      console.warn(
        `Project data file missing at ${PROJECTS_DATA_PATH}. Did you run the GitHub Action?`,
      );
      return [];
    }
    const data = JSON.parse(fs.readFileSync(PROJECTS_DATA_PATH, "utf8"));
    return Array.isArray(data.projects) ? data.projects : [];
  } catch (error) {
    console.warn("Could not read projects-data.json:", error);
    return [];
  }
}

/**
 * This is the single source of truth for all project data in the application.
 * It orchestrates the merging of three distinct data sources:
 *
 * 1. Raw Data (`public/projects-data.json`): The base project list, generated
 *    by the GitHub Actions script from the API and manual-projects-data.json.
 *
 * 2. Content Data (`src/content/projects-extra/`): Long-form Markdown content
 *    for specific projects, linked by slug (convention-based).
 *
 * 3. Behavioral Config (`src/config/site.ts`): Explicit overrides
 *    for presentation (artwork, highlights, assets).
 *
 * Any component or page needing project data should use this function.
 */
export async function getAllProjects(): Promise<ProjectRecord[]> {
  // 1. Load the raw project data from the canonical JSON file.
  const rawProjects = readProjectsFile();

  // 2. Load the extra content data from the content collection.
  const extraContentEntries = await getCollection("projectsExtra");
  const extraContentMap = new Map(
    extraContentEntries.map((entry) => [entry.data.slug, entry]),
  );

  // 3. Combine all sources into a unified project record.
  return rawProjects.map((project: any) => {
    const slug = project.slug || slugify(project.name);
    const config = projectsConfig.config[slug] ?? {};

    return {
      name: project.name,
      homepage: project.homepage,
      description: project.description,
      githubUrl: project.githubUrl ?? null,
      slug,
      pubDate: project.pubDate,
      updatedDate: project.updatedDate,
      previewAsset: config.previewAsset ?? null,
      shouldShowPreview: config.shouldShowPreview ?? false,
      previewType: (config.previewType ?? "titlecard") as PreviewArtworkType,
      isHighlight: config.isHighlight ?? false,
      highlightOrder: config.order ?? 999,
      hasExtraContent: extraContentMap.has(slug),
    } satisfies ProjectRecord;
  });
}

export async function getHighlightedProjects(): Promise<ProjectRecord[]> {
  const projects = await getAllProjects();
  return projects
    .filter((project) => project.isHighlight)
    .sort((a, b) => a.highlightOrder - b.highlightOrder);
}
