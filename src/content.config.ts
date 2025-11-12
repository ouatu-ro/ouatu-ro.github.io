import { defineCollection, z } from "astro:content";
import { blog as blogConfig, type CategoryId } from "./config/site";
import { glob } from "astro/loaders";

const categoryKeys = Object.keys(blogConfig.categories) as [
  CategoryId,
  ...CategoryId[],
];

const blog = defineCollection({
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
    slug: z.string(),
  }),
});

export const collections = { blog, projectsExtra };
