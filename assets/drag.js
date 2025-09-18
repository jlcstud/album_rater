window.addEventListener('load', function() {
    const observer = new MutationObserver((mutations, obs) => {
        // Look for the specific element that Plotly creates.
        const plotlyNode = document.querySelector('.js-plotly-plot');
        
        if (plotlyNode) {
            // Check if we've already initialized this node.
            if (!plotlyNode._dragRatingInitialized) {
                plotlyNode._dragRatingInitialized = true;
                initializeDragRating(plotlyNode);
            }
            // Once we find it, we don't need to keep observing.
            obs.disconnect();
        }
    });

    // Start observing the body for changes.
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});

function initializeDragRating(graphNode) {
    const gd = graphNode;
    let isDragging = false;
    let dragIndex = -1;
    let albumId = '';

    try {
        // The ID is on the parent of the .js-plotly-plot element
        const idObj = JSON.parse(gd.parentElement.id);
        albumId = idObj.album;
    } catch (e) {
        console.error('Could not parse album ID from graph element:', e);
        return;
    }

    if (!albumId) {
        console.error('Album ID not found in graph element.');
        return;
    }

    gd.on('plotly_mousedown', (e) => {
        const point = e.points[0];
        const barIndex = point.pointNumber;
        const isIgnored = gd.data[0].customdata[barIndex][1];

        if (isIgnored) return;

        isDragging = true;
        dragIndex = barIndex;
        e.event.preventDefault();
        updateRatingFromEvent(e.event);
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        updateRatingFromEvent(e);
    });

    window.addEventListener('mouseup', () => {
        if (!isDragging) return;
        isDragging = false;
        dragIndex = -1;
    });

    function updateRatingFromEvent(event) {
        const plotArea = gd.querySelector('.plot');
        const plotRect = plotArea.getBoundingClientRect();
        const yPixel = event.clientY - plotRect.top;

        let rating = 10 * (1 - (yPixel / plotRect.height));
        rating = Math.max(0, Math.min(10.0, rating));
        rating = Math.round(rating * 10) / 10;

        const newY = [...gd.data[0].y];
        newY[dragIndex] = rating;
        Plotly.restyle(gd, 'y', [newY]);

        sendRatingUpdate(albumId, dragIndex, rating);
    }

    let debounceTimer;
    function sendRatingUpdate(albumId, trackIndex, rating) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            const storeNode = document.querySelector(`[id*='"type":"album-state","album":"${albumId}"']`);
            if (!storeNode) {
                console.error('Could not find album-state store.');
                return;
            }
            const state = JSON.parse(storeNode.textContent);
            const ratings = state.ratings;
            ratings[trackIndex] = rating;

            fetch('/api/rate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    album_id: albumId,
                    ratings: ratings,
                    ignored: state.ignored
                }),
            }).catch(console.error);
        }, 250);
    }
}
