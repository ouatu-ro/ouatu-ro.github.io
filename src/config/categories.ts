export const CATEGORIES = {
  essay: {
    name: "Essay",
    description: "Long-form writing, reflections, and thoughts on technology.",
  },
  lab: {
    name: "Tech",
    description: "In-depth technical explorations and code deep-dives.",
  },
  // catalogue: {
  //   name: "Catalogue",
  //   description: "Curated lists, resources, and structured collections.",
  // },
  note: {
    name: "Note",
    description: "Crief explorations, fragments, and conceptual sketches"
  }
} as const;

export type CategoryId = keyof typeof CATEGORIES;

export const CATEGORY_IDS = Object.keys(CATEGORIES) as CategoryId[];

export function isCategory(value: string): value is CategoryId {
  return Object.prototype.hasOwnProperty.call(CATEGORIES, value);
}
