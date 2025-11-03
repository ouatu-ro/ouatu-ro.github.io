import { glob } from "astro/loaders";
import { defineCollection, z } from "astro:content";

const labs = defineCollection({
  loader: glob({ base: "./src/content/labs", pattern: "**/*.{md,mdx}" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    heroImage: z.string().optional(),
    math: z.boolean().optional(),
  }),
});

const essays = defineCollection({
  loader: glob({ base: "./src/content/essays", pattern: "**/*.{md,mdx}" }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    pubDate: z.coerce.date().optional(),
    updatedDate: z.coerce.date().optional(),
    heroImage: z.string().optional(),
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

export const collections = { labs, essays, projectsExtra };
