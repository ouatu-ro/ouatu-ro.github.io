export const CATEGORIES = {
  lab: {
    name: "Lab",
    description: "In-depth technical explorations and code deep-dives.",
  },
  // essay: {
  //   name: "Essay",
  //   description: "Long-form writing, reflections, and thoughts on technology.",
  // },
  // catalogue: {
  //   name: "Catalogue",
  //   description: "Curated lists, resources, and structured collections.",
  // },
  note: {
    name: "Note",
    description: "Short notes"
  }
} as const;

export type CategoryId = keyof typeof CATEGORIES;

export const CATEGORY_IDS = Object.keys(CATEGORIES) as CategoryId[];

export function isCategory(value: string): value is CategoryId {
  return Object.prototype.hasOwnProperty.call(CATEGORIES, value);
}
