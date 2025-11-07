import type { CollectionEntry } from "astro:content";
import { getCollection } from "astro:content";
import { CATEGORIES, type CategoryId } from "../config/categories";
import { WORDS_PER_MINUTE } from "../config/constants";
import { slugify } from "../utils/slugify";

export type BlogCollectionEntry = CollectionEntry<"blog">;

export type BlogPost = BlogCollectionEntry & {
  slug: string;
  readingTime: number;
  normalizedTags: TagSummary[];
};

export type TagSummary = {
  label: string;
  slug: string;
  count?: number;
};

const wordsRegex = /\s+/g;

const toSortDate = (entry: BlogCollectionEntry) =>
  entry.data.updatedDate ?? entry.data.pubDate;

export const normalizeTag = (tag: string) =>
  slugify(tag).replace(/^-+|-+$/g, "");

const deriveSlugFromId = (id: string) =>
  id.replace(/\\/g, "/").replace(/\.mdx?$/, "");

const computePostProperties = (post: BlogCollectionEntry): BlogPost => {
  const slug = post.data.slug ?? deriveSlugFromId(post.id);
  const source = post.body ?? "";
  const words = source.split(wordsRegex).filter(Boolean).length;
  const readingTimeRaw =
    post.data.readingTime ?? Math.ceil(words / WORDS_PER_MINUTE);
  const readingTime = Math.max(1, readingTimeRaw || 1);
  const normalizedTags = (post.data.tags ?? []).map((tag) => ({
    label: tag,
    slug: normalizeTag(tag),
  }));
  const enhanced = post as BlogPost;
  return {
    ...enhanced,
    slug,
    readingTime,
    normalizedTags,
  };
};

const summarizeTags = (posts: BlogPost[]): TagSummary[] => {
  const counts = new Map<string, TagSummary>();
  for (const post of posts) {
    for (const tag of post.normalizedTags) {
      const existing = counts.get(tag.slug);
      if (existing) {
        existing.count = (existing.count ?? 0) + 1;
      } else {
        counts.set(tag.slug, { ...tag, count: 1 });
      }
    }
  }
  return Array.from(counts.values()).sort(
    (a, b) => (b.count ?? 0) - (a.count ?? 0),
  );
};

export async function getAllPosts(): Promise<BlogPost[]> {
  const posts = await getCollection("blog");
  return posts
    .filter((post) => !post.data.draft)
    .map(computePostProperties)
    .sort((a, b) => toSortDate(b).valueOf() - toSortDate(a).valueOf());
}

export async function getPostsByCategory(category: CategoryId): Promise<BlogPost[]> {
  const posts = await getAllPosts();
  return posts.filter((post) => post.data.category === category);
}

export async function getPostsByTag(
  tagSlug: string,
  category?: CategoryId,
): Promise<BlogPost[]> {
  const posts = await getAllPosts();
  const normalized = normalizeTag(tagSlug);
  return posts.filter(
    (post) =>
      (!category || post.data.category === category) &&
      post.normalizedTags.some((tag) => tag.slug === normalized),
  );
}

export async function getAllTags(category?: CategoryId): Promise<TagSummary[]> {
  const posts = category
    ? await getPostsByCategory(category)
    : await getAllPosts();
  return summarizeTags(posts);
}

export function getCategoryMeta(category: CategoryId) {
  return CATEGORIES[category];
}

export { deriveSlugFromId };
