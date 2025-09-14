// Debug script to help diagnose search and navigation issues
(function() {
  // Add event logging for search result clicks
  document.addEventListener('click', function(e) {
    // Find if the click target is within a search result
    const searchResult = e.target.closest('a[id^=\'{"type":"search-result"\']');
    if (searchResult) {
      console.log('Search result clicked:', searchResult);
      console.log('Album ID:', searchResult.getAttribute('id'));
      console.log('Href:', searchResult.getAttribute('href'));
      
      // Force the navigation to work by preventing default and manually setting location
      e.preventDefault();
      const href = searchResult.getAttribute('href');
      console.log('Forcing navigation to:', href);
      window.location.pathname = href;
    }
  });

  // Monitor URL changes
  let lastPathname = window.location.pathname;
  setInterval(() => {
    if (window.location.pathname !== lastPathname) {
      console.log('URL changed from', lastPathname, 'to', window.location.pathname);
      lastPathname = window.location.pathname;
    }
  }, 100);

  console.log('Debug script loaded');
})();