// Function to convert project name to slug
function slugify(text) {
  return text
    .toString()
    .toLowerCase()
    .replace(/\s+/g, '-')       // Replace spaces with -
    .replace(/[^\w\-]+/g, '')   // Remove all non-word chars
    .replace(/\-\-+/g, '-')     // Replace multiple - with single -
    .trim();                    // Trim - from start and end
}

// Make it available for both browser and Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { slugify };
}
