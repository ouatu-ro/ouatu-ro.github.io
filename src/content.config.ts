import { glob } from "astro/loaders";
import { defineCollection, z } from "astro:content";

const blog = defineCollection({
  loader: glob({ base: "./src/content/blog", pattern: "**/*.{md,mdx}" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    heroImage: z.string().optional(),
    math: z.boolean().optional(),
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
  }),
});

export const collections = { blog, projectsExtra };
