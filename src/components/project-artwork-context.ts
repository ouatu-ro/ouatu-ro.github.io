import TerminalArtwork from "./project-artworks/TerminalArtwork.astro";
import WireframeArtwork from "./project-artworks/WireframeArtwork.astro";
import GlyphArtwork from "./project-artworks/GlyphArtwork.astro";
import BlueprintArtwork from "./project-artworks/BlueprintArtwork.astro";
import CodeArtwork from "./project-artworks/CodeArtwork.astro";
import GraphArtwork from "./project-artworks/GraphArtwork.astro";
import MeshArtwork from "./project-artworks/MeshArtwork.astro";
import OuroborosArtwork from "./project-artworks/OuroborosArtwork.astro";
import ScreenshotArtwork from "./project-artworks/ScreenshotArtwork.astro";
import TitleCardArtwork from "./project-artworks/TitleCardArtwork.astro";
import WinkArtwork from "./project-artworks/WinkArtwork.astro";
import type { ProjectRecord } from "../utils/projects";

type Pos = { top: string; left: string };
type Connector = Pos & { width: string; rotate: string };

const blueprintNodes: Pos[] = [
  { top: "12%", left: "18%" },
  { top: "25%", left: "68%" },
  { top: "58%", left: "32%" },
  { top: "70%", left: "74%" },
  { top: "40%", left: "48%" },
];

const blueprintConnectors: Connector[] = [
  { top: "20%", left: "20%", width: "52%", rotate: "12deg" },
  { top: "42%", left: "30%", width: "44%", rotate: "-6deg" },
  { top: "62%", left: "18%", width: "62%", rotate: "10deg" },
];

const graphNodes: Pos[] = [
  { top: "18%", left: "30%" },
  { top: "42%", left: "58%" },
  { top: "65%", left: "24%" },
  { top: "30%", left: "76%" },
  { top: "68%", left: "68%" },
];

const graphLinks: Connector[] = [
  { top: "26%", left: "32%", width: "30%", rotate: "18deg" },
  { top: "50%", left: "44%", width: "26%", rotate: "-12deg" },
  { top: "60%", left: "26%", width: "48%", rotate: "8deg" },
];

export const previewComponentMap = {
  terminal: TerminalArtwork,
  wireframe: WireframeArtwork,
  glyph: GlyphArtwork,
  blueprint: BlueprintArtwork,
  code: CodeArtwork,
  graph: GraphArtwork,
  mesh: MeshArtwork,
  ouroboros: OuroborosArtwork,
  screenshot: ScreenshotArtwork,
  titlecard: TitleCardArtwork,
  wink: WinkArtwork,
} as const;

export type PreviewArtworkType = keyof typeof previewComponentMap;

export interface ProjectArtworkContext {
  previewType: PreviewArtworkType;
  promptName: string;
  pascalName: string;
  trimmedDescription: string;
  terminalLines: string[];
  codeLines: string[];
  blueprintNodes: Pos[];
  blueprintConnectors: Connector[];
  graphNodes: Pos[];
  graphLinks: Connector[];
}

export function buildProjectArtworkContext(
  project: Pick<ProjectRecord, "name" | "slug" | "description" | "previewType">,
): ProjectArtworkContext {
  const rawDescription = project.description ?? "";
  const firstLine = rawDescription.split(/\r?\n/)[0] ?? "";
  const maxLength = 150;
  const trimmedDescription =
    firstLine.length > maxLength
      ? `${firstLine.slice(0, maxLength - 1).trimEnd()}…`
      : firstLine;

  const normalizedSlug = (project.slug ?? project.name ?? "")
    .replace(/[^a-z0-9]+/gi, "-");

  const promptName = normalizedSlug.replace(/-([a-z])/g, (_, char) =>
    char.toUpperCase()
  );

  const pascalName = promptName
    .replace(/(?:^|-)([a-z0-9])/g, (_, char) => char.toUpperCase())
    .replace(/-/g, "");

  const previewType =
    (project.previewType ?? "titlecard") as PreviewArtworkType;

  return {
    previewType,
    promptName,
    pascalName,
    trimmedDescription,
    terminalLines: [
      `> launch ${promptName}`,
      "# compiling modules… ok",
      "# running simulation…",
      "# solved in 0.13s",
    ],
    codeLines: [
      `export function ${pascalName}() {`,
      `  const app = initTool({ name: "${project.name}" });`,
      "  return app.run();",
      "}",
    ],
    blueprintNodes,
    blueprintConnectors,
    graphNodes,
    graphLinks,
  };
}
