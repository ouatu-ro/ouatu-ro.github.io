/**
 * Convert text to a URL-friendly slug
 * This function is used on both the server side and client side
 */
export function slugify(text: string): string {
  return text
    .toString()
    .toLowerCase()
    .replace(/\s+/g, "-") // Replace spaces with -
    .replace(/[^\w\-]+/g, "") // Remove all non-word chars
    .replace(/\-\-+/g, "-") // Replace multiple - with single -
    .trim(); // Trim - from start and end
}

// Make the slugify function available in the global scope when loaded in browser
// This supports pages that use the function directly in inline scripts
if (typeof window !== "undefined") {
  (window as any).slugify = slugify;
}
