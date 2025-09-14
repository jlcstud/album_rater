// search-helper.js - Enhanced behavior for search modal
(function() {
  console.log('Search helper script v3 loaded - relaxed click handling');
  
  // Variables to prevent auto-clicks
  let modalJustOpened = false;
  
  // Helper to ensure clicks on search result cards navigate properly
  document.addEventListener('click', function(e) {
    // Brief delay to prevent immediate auto-clicks
    if (modalJustOpened) {
      console.log('Ignoring potential auto-click (modal just opened)');
      return;
    }
    
    // Find if the click target is within a search result card
    let cardElement = e.target.closest('.search-result-item');
    if (cardElement) {
      // Find the album ID from the card element ID
      const cardId = cardElement.id;
      if (cardId && cardId.includes('album_id')) {
        try {
          // Extract album ID from the JSON-formatted ID
          const idObj = JSON.parse(cardId.replace(/^{"type":"search-result-card",/, '{"type":"search-result-card",'));
          const albumId = idObj.album_id;
          
          if (albumId) {
            console.log('Search result clicked for album:', albumId);
            // Force navigation to album page
            window.location.href = `/album/${albumId}`;
            return false;
          }
        } catch (e) {
          console.error('Error parsing card ID:', e);
        }
      }
    }
  });
  
  // Add visual cue for clickable cards
  function enhanceSearchCards() {
    const cards = document.querySelectorAll('.search-result-item');
    cards.forEach(card => {
      card.style.cursor = 'pointer';
    });
    
    // Brief protection against immediate auto-clicks
    modalJustOpened = true;
    setTimeout(() => {
      modalJustOpened = false;
      console.log('Search result clicks fully enabled');
    }, 200); // Much shorter delay
  }
  
  // Setup observer to watch for modal opening
  const observer = new MutationObserver(function(mutations) {
    for (const mutation of mutations) {
      if (mutation.type === 'attributes' && 
          mutation.attributeName === 'class' && 
          mutation.target.classList.contains('show')) {
        enhanceSearchCards();
      }
    }
  });
  
  // Start observing modal
  setTimeout(() => {
    const modal = document.querySelector('.modal');
    if (modal) {
      observer.observe(modal, { attributes: true });
      console.log('Search modal observer activated');
    }
  }, 1000);
  
  window.searchHelperLoaded = true;
})();