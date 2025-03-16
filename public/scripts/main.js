// Function to blink a section when navigated to
function blinkSection(sectionId) {
  const section = document.getElementById(sectionId);
  const originalBgColor = section.style.backgroundColor;
  section.style.backgroundColor = "#454545";
  setTimeout(() => (section.style.backgroundColor = originalBgColor), 500);
}
window.blinkSection = blinkSection;

// Tooltip functionality
function showTooltip(event, text) {
  const tooltip = document.getElementById("tooltip");
  tooltip.innerHTML = text;
  tooltip.style.left = event.pageX + 15 + "px";
  tooltip.style.top = event.pageY + 15 + "px";
  tooltip.style.opacity = 1;
}
function moveTooltip(event) {
  const tooltip = document.getElementById("tooltip");
  tooltip.style.left = event.pageX + 15 + "px";
  tooltip.style.top = event.pageY + 15 + "px";
}
function hideTooltip() {
  const tooltip = document.getElementById("tooltip");
  tooltip.style.opacity = 0;
}

// Initialize project descriptions and tooltips
function initializeProjectInteractions() {
  const infoButtons = document.querySelectorAll(".info-btn");

  infoButtons.forEach((infoBtn) => {
    const item = infoBtn.closest("li");
    const descriptionEl = item.querySelector(".project-description");
    const description = descriptionEl ? descriptionEl.textContent : "";

    // Desktop tooltip on info button
    infoBtn.addEventListener("mouseenter", (event) =>
      showTooltip(event, description)
    );
    infoBtn.addEventListener("mousemove", moveTooltip);
    infoBtn.addEventListener("mouseleave", hideTooltip);

    // Toggle description on info button click
    infoBtn.addEventListener("click", (e) => {
      e.preventDefault();
      const wasActive = descriptionEl.classList.contains("active");
      // Hide all other active descriptions
      document.querySelectorAll(".project-description.active").forEach((el) => {
        el.classList.remove("active");
      });
      if (!wasActive) {
        descriptionEl.classList.add("active");
        // Add a one-time click event to the document to close description
        setTimeout(() => {
          const closeHandler = (event) => {
            if (!item.contains(event.target)) {
              descriptionEl.classList.remove("active");
              document.removeEventListener("click", closeHandler);
            }
          };
          document.addEventListener("click", closeHandler);
        }, 0);
      }
    });
  });
}

// Initialize project interactions on page load and after navigation
document.addEventListener("DOMContentLoaded", initializeProjectInteractions);
document.addEventListener("astro:after-swap", initializeProjectInteractions);

// Call immediately in case the DOM is already loaded
if (
  document.readyState === "complete" ||
  document.readyState === "interactive"
) {
  setTimeout(initializeProjectInteractions, 1);
}
