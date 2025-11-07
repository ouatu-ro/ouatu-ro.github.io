import { glob } from "astro/loaders";
import { defineCollection, z } from "astro:content";
import { CATEGORIES, type CategoryId } from "./config/categories";

const categoryKeys = Object.keys(CATEGORIES) as [
  CategoryId,
  ...CategoryId[],
];

const blog = defineCollection({
  loader: glob({ base: "./src/content/blog", pattern: "**/*.{md,mdx}" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    slug: z.string().optional(),
    category: z.enum(categoryKeys),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    heroImage: z.string().optional(),
    ogImage: z.string().url().optional(),
    math: z.boolean().optional(),
    readingTime: z.number().optional(),
    canonicalUrl: z.string().url().optional(),
  }),
});

const projectsExtra = defineCollection({
  loader: glob({
    base: "./src/content/projects-extra",
    pattern: "**/*.{md,mdx}",
  }),
  schema: z.object({
    slug: z.string(), // must match project.slug from projects-data.json
    title: z.string().optional(), // optional override
    order: z.number().optional(),
    projectPreview: z.string().optional(),
    shouldShowPreview: z.boolean().optional(),
  }),
});

const highlights = defineCollection({
  type: "data",
  schema: z.object({
    projects: z
      .array(
        z.object({
          slug: z.string(),
          previewType: z
            .enum([
              "terminal",
              "wireframe",
              "glyph",
              "blueprint",
              "screenshot",
              "graph",
              "code",
              "mesh",
            ])
            .optional(),
        }),
      )
      .default([]),
    posts: z
      .array(
        z.object({
          slug: z.string(),
        }),
      )
      .default([]),
  }),
});

export const collections = { blog, projectsExtra, highlights };
