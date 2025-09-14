// Debug script to help diagnose search and navigation issues
(function() {
  console.log('Debug script v2 loaded - click tracking enabled');
  
  // Track all clicks
  document.addEventListener('click', function(e) {
    const target = e.target;
    const closestCard = target.closest('.search-result-item');
    
    console.log('Click detected on:', target.tagName, target.className);
    if (closestCard) {
      console.log('Within search card:', closestCard.id);
    }
  }, true);
  
  // Debug modal events
  const debugModalEvents = function() {
    const modal = document.querySelector('.modal');
    if (modal) {
      console.log('Found modal, adding event listeners');
      
      modal.addEventListener('shown.bs.modal', function() {
        console.log('Modal shown event fired');
      });
      
      modal.addEventListener('hidden.bs.modal', function() {
        console.log('Modal hidden event fired');
      });
    } else {
      setTimeout(debugModalEvents, 1000);
    }
  };
  
  // Start watching modal events
  debugModalEvents();
  
  // Global helper function to disable auto-navigation
  window.disableAutoNav = function() {
    console.log('Disabling auto-navigation...');
    
    // Force all search result cards to ignore first click
    const cards = document.querySelectorAll('.search-result-item');
    cards.forEach(card => {
      card.setAttribute('data-first-click-ignored', 'false');
      
      // Add protection to ignore the first click on each card
      card.addEventListener('click', function(e) {
        if (card.getAttribute('data-first-click-ignored') === 'false') {
          console.log('Ignoring first click on card', card.id);
          card.setAttribute('data-first-click-ignored', 'true');
          e.preventDefault();
          e.stopPropagation();
          return false;
        }
      }, true);
    });
  };
  
  // Enable this after a delay
  setTimeout(function() {
    console.log('Debug functions available - use window.disableAutoNav() to prevent auto-navigation');
  }, 2000);
})();