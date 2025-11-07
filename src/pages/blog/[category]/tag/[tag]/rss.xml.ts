import type { APIContext } from "astro";
import {
  CATEGORY_IDS,
  CATEGORIES,
  type CategoryId,
} from "@/config/categories";
import { createRssResponse } from "@/lib/feeds";
import { getAllTags, getPostsByTag } from "@/lib/posts";

export async function getStaticPaths() {
  const paths = [];
  for (const category of CATEGORY_IDS) {
    const tags = await getAllTags(category);
    for (const tag of tags) {
      const posts = await getPostsByTag(tag.slug, category);
      if (posts.length === 0) continue;
      paths.push({
        params: { category, tag: tag.slug },
        props: { category, tagLabel: tag.label },
      });
    }
  }
  return paths;
}

export async function GET(context: APIContext) {
  const category = context.params.category as CategoryId;
  const tagSlug = context.params.tag as string;
  const posts = await getPostsByTag(tagSlug, category);
  const label = context.props?.tagLabel ?? tagSlug;
  const categoryMeta = CATEGORIES[category];
  return createRssResponse({
    context,
    posts,
    selfPath: `/blog/${category}/tag/${tagSlug}/rss.xml`,
    title: `${categoryMeta.name} · #${label}`,
    description: `Posts tagged ${label} in ${categoryMeta.name}`,
  });
}
