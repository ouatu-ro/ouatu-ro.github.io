// Enhanced Burger Menu
function setupBurgerMenu() {
  const burgerMenu = document.querySelector<HTMLElement>(".burger-menu");
  const navLinks = document.querySelector<HTMLElement>(".nav-links");

  if (burgerMenu && navLinks) {
    burgerMenu.addEventListener("click", () => {
      burgerMenu.classList.toggle("active");
      navLinks.classList.toggle("active");
    });

    // Close menu when clicking elsewhere
    document.addEventListener("click", (e) => {
      const target = e.target as HTMLElement;
      if (
        !target.closest(".burger-menu") &&
        !target.closest(".nav-links") &&
        navLinks.classList.contains("active")
      ) {
        burgerMenu.classList.remove("active");
        navLinks.classList.remove("active");
      }
    });
  }
}

// Add copy buttons to code blocks with data-language attribute
function addCopyButtonsToCodeBlocks() {
  const codeBlocks = document.querySelectorAll<HTMLElement>(
    'pre[data-language]:not([data-language="plaintext"])'
  );

  codeBlocks.forEach((block) => {
    // Create copy button
    const copyButton = document.createElement("button");
    copyButton.className = "copy-button";
    copyButton.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy';

    // Add click event handler
    copyButton.addEventListener("click", () => {
      // Get the code content
      const code = block.querySelector("code");
      let textToCopy = "";

      if (code) {
        // Get all lines and remove line numbers
        const lines = Array.from(code.querySelectorAll(".line"));
        textToCopy = lines.map((line) => line.textContent).join("\n");
      } else {
        // Fallback to innerText if .line elements aren't found
        textToCopy = block.innerText;
      }

      // Copy to clipboard
      navigator.clipboard
        .writeText(textToCopy)
        .then(() => {
          // Show success state
          copyButton.classList.add("copied");
          copyButton.innerHTML =
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!';

          // Reset after 2 seconds
          setTimeout(() => {
            copyButton.classList.remove("copied");
            copyButton.innerHTML =
              '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy';
          }, 2000);
        })
        .catch((err) => {
          console.error("Failed to copy code: ", err);
          copyButton.textContent = "Error!";
        });
    });

    // Add button to the code block
    block.appendChild(copyButton);
  });
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  setupBurgerMenu();
  addCopyButtonsToCodeBlocks();
});

// Export functions for importing in components if needed
export { setupBurgerMenu, addCopyButtonsToCodeBlocks };
