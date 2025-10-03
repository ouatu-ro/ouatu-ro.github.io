/** @type {import('stylelint').Config} */
export default {
  extends: ["stylelint-config-standard", "stylelint-config-recommended"],
  plugins: ["stylelint-order"],
  rules: {
    "no-duplicate-selectors": true,
    "declaration-block-no-duplicate-properties": true,
    "import-notation": "url",

    // Relaxed rules to suit componentized CSS approach
    "selector-class-pattern": null,
    "declaration-no-important": null,
    "order/properties-alphabetical-order": null,
    "font-family-name-quotes": null,
    "alpha-value-notation": null,
    "color-function-notation": null,
    "color-function-alias-notation": null,
    "media-feature-range-notation": null,
    "no-descending-specificity": null,
    "shorthand-property-no-redundant-values": null,
    "declaration-block-no-redundant-longhand-properties": null,
    "length-zero-no-unit": null,
    "keyframes-name-pattern": null,
    "rule-empty-line-before": null,
    "at-rule-empty-line-before": null,
    "color-hex-length": "short",
    "comment-empty-line-before": null,
  },
};
