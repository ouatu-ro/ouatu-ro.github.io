// --- PROJECTS CONFIG ---
import type { HighlightPreviewType } from "../lib/projects";

// --- SITE METADATA ---
export const site = {
  title: "Bogdan Ouatu",
  description:
    "Personal site by Bogdan Ouatu — essays, labs, and projects on systems, and machine learning.",
  author: "Bogdan Ouatu",
  authorTitle: "Machine Learning Engineer",
  tagline: "Joy-Driven Programming",
  email: "hello@ouatu.ro",
  githubUrl: "https://github.com/ouatu-ro/",
  linkedinUrl: "https://www.linkedin.com/in/bogdan-ouatu/",
  kaggleUrl: "https://www.kaggle.com/ilikehaskell/discussion/",
  gaTrackingId: "G-FWXVQ3N6SL",
} as const;

// --- BLOG CONFIG ---
export const blog = {
  postsPerPage: 10,
  wordsPerMinute: 200,
  categories: {
    essay: {
      name: "Essay",
      description:
        "Long-form writing, reflections, and thoughts on technology.",
    },
    lab: {
      name: "Tech",
      description: "In-depth technical explorations and code deep-dives.",
    },
    note: {
      name: "Note",
      description: "Brief explorations, fragments, and conceptual sketches.",
    },
  },
  highlightedPosts: [{ slug: "common-sense-learning" }, { slug: "als" }],
} as const;

// --- PROJECTS CONFIG ---
export const projects = {
  config: {
    "maze-solver": {
      previewType: "terminal",
      isHighlight: true,
      order: 1,
      previewAsset: "/preview/maze-demo-720p.webm",
      shouldShowPreview: true,
    },
    "table-planner": { previewType: "wireframe", isHighlight: true, order: 2 },
    "hall-of-mirrors-3": { previewType: "mesh", isHighlight: true, order: 3 },
    "this-website": { previewType: "ouroboros" },
    verbalate: { previewType: "graph" },
    "wink-sound-effects": { previewType: "wink" },
  } as Record<
    string,
    {
      previewType?: HighlightPreviewType;
      isHighlight?: boolean;
      order?: number;
      previewAsset?: string;
      shouldShowPreview?: boolean;
    }
  >,
} as const;

export type CategoryId = keyof typeof blog.categories;
