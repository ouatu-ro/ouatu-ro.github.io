/**
 * @file Central registry for project preview artwork components.
 *
 * Each key in `previewComponentMap` corresponds to a `previewType` value stored
 * on a `ProjectRecord`. The value is the Astro component responsible for
 * rendering that artwork.
 *
 * To add a new artwork:
 * 1. Create the component under `src/components/project-artworks/`.
 * 2. Import it here.
 * 3. Add a new key/value entry to `previewComponentMap`.
 *
 * The `PreviewArtworkType` union is derived from the map's keys, providing a
 * single source of truth and type safety throughout the app.
 */
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
