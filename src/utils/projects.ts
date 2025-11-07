import fs from "node:fs";
import path from "node:path";
import { getCollection, getEntry } from "astro:content";
import { slugify } from "./slugify";

export interface ProjectRecord {
  name: string;
  homepage: string;
  description: string;
  githubUrl: string | null;
  slug: string;
  pubDate?: string;
  updatedDate?: string;
  previewAsset?: string | null;
  shouldShowPreview?: boolean;
  previewType?: HighlightPreviewType;
}

export type HighlightPreviewType =
  | "terminal"
  | "wireframe"
  | "glyph"
  | "blueprint"
  | "screenshot"
  | "graph"
  | "code"
  | "mesh"
  | "ouroboros"
  | "titlecard"
  | "wink";

export interface HighlightsConfig {
  projects: Array<{ slug: string; previewType?: HighlightPreviewType }>;
  posts: Array<{ slug: string }>;
}

const PROJECTS_FILE = path.join(
  process.cwd(),
  "public",
  "projects-data.json"
);
const MANUAL_PROJECTS_FILE = path.join(
  process.cwd(),
  "public",
  "manual-projects-data.json"
);

export function readProjectsData() {
  try {
    if (fs.existsSync(PROJECTS_FILE)) {
      const data = JSON.parse(fs.readFileSync(PROJECTS_FILE, "utf8"));
      return (data.projects ?? []).map((project: Record<string, unknown>) => ({
        ...project,
      }));
    }
  } catch (error) {
    console.warn("Could not read projects-data.json:", error);
  }

  try {
    if (fs.existsSync(MANUAL_PROJECTS_FILE)) {
      const data = JSON.parse(fs.readFileSync(MANUAL_PROJECTS_FILE, "utf8"));
      return data.manualProjects ?? [];
    }
  } catch (error) {
    console.warn("Could not read manual-projects-data.json:", error);
  }

  console.error(
    "No projects found! Please ensure the GitHub Action has generated the projects-data.json file."
  );
  return [];
}

export async function getHighlightConfig(): Promise<HighlightsConfig> {
  try {
    const entry = await getEntry("highlights", "site");
    if (!entry) {
      console.warn("Highlights configuration entry missing, using defaults.");
      return { projects: [], posts: [] };
    }
    return entry.data as HighlightsConfig;
  } catch (error) {
    console.warn("Highlights configuration missing, using defaults.", error);
    return { projects: [], posts: [] };
  }
}

export async function getProjects(): Promise<ProjectRecord[]> {
  const baseProjects = readProjectsData();
  const extras = await getCollection("projectsExtra");
  const highlightConfig = await getHighlightConfig();
  const previewTypeMap = new Map(
    highlightConfig.projects.map((item) => [item.slug, item.previewType])
  );

  const defaultPreviewTypes: Record<string, HighlightPreviewType> = {
    "this-website": "ouroboros",
    verbalate: "graph",
    "maze-solver": "terminal",
    "table-planner": "wireframe",
    "hall-of-mirrors-3": "mesh",
    "wink-sound-effects": "wink",
  };

  return baseProjects.map((project: any) => {
    const slug = project.slug || slugify(project.name);
    const extra = extras.find((entry) => entry.data.slug === slug);
    const defaultType = defaultPreviewTypes[slug] ?? defaultPreviewTypes[slugify(project.name)];

    return {
      name: project.name,
      homepage: project.homepage,
      description: project.description,
      githubUrl: project.githubUrl ?? null,
      slug,
      pubDate: project.pubDate,
      updatedDate: project.updatedDate,
      previewAsset: extra?.data.projectPreview ?? null,
      shouldShowPreview: extra?.data.shouldShowPreview ?? false,
      previewType:
        previewTypeMap.get(slug) ??
        previewTypeMap.get(slugify(project.name)) ??
        defaultType ??
        "titlecard",
    } satisfies ProjectRecord;
  });
}

export async function getHighlightedProjects() {
  const [projects, highlights] = await Promise.all([
    getProjects(),
    getHighlightConfig(),
  ]);

  const projectMap = new Map(projects.map((project) => [project.slug, project]));

  return highlights.projects
    .map((ref) => {
      const project = projectMap.get(ref.slug);
      if (!project) return null;
      return {
        ...project,
        previewType: ref.previewType ?? project.previewType,
      };
    })
    .filter(Boolean) as ProjectRecord[];
}

export type HighlightEntries = Awaited<ReturnType<typeof getHighlightConfig>>;
