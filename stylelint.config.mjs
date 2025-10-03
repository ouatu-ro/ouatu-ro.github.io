/** @type {import('stylelint').Config} */
export default {
  extends: ["stylelint-config-standard", "stylelint-config-recommended"],
  rules: {
    // Core hygiene
    "no-duplicate-selectors": true,
    "declaration-no-important": true,
    "declaration-block-no-duplicate-properties": true,

    // Naming convention — keep simple, less regex churn
    "selector-class-pattern":
      "^(c|p|l|u|blog|project|seo|header|footer|nav|btn)-[a-z0-9-]+$",

    // Optional: mild ordering without overkill
    "order/properties-alphabetical-order": true,
  },
};
