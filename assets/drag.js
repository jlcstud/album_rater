// drag.js - custom drag-to-rate logic for album rating graph
// This script attaches mouse handlers to Plotly graphs with id pattern album-graph
// and performs live bar updates + POST to /api/rate on mouseup.
(function(){
  const DEBUG = true; // set false to silence logs
  function log(){ if(DEBUG) console.log('[drag-rate]', ...arguments); }
  function snap(val){ return Math.max(0, Math.min(10, Math.round(val*10)/10)); }
  function findGraphDivs(){
    return Array.from(document.querySelectorAll('div.js-plotly-plot'))
      .filter(d=>{ try { return d.layout && d.layout.uirevision; } catch(e){ return false; } });
  }
  let active = null; // {div, album_id, ratings, ignored, downAt}
  let isDown = false;
  let lastClickTime = 0;

  function init(div){
    if(div.__dragRatingsInit) return; div.__dragRatingsInit = true;
    log('Init graph', div.layout && div.layout.uirevision);
    div.addEventListener('mousedown', (e)=>{
      if(e.button!==0) return;
      const info = collectState(div);
      if(!info) return;
      active = info; isDown = true; active.downAt = Date.now();
      handlePos(e, info, true);
    });
  }

  window.addEventListener('mousemove', (e)=>{ if(isDown && active) handlePos(e, active, false); });
  window.addEventListener('mouseup', (e)=>{
    if(!isDown || !active) return;
    isDown = false;
    // treat short press without movement as click; already handled by handlePos
    persist(active);
    active = null;
  });

  function collectState(gd){
    if(!gd || !gd.data || !gd.data.length) return null;
    const barTrace = gd.data[0];
    if(!barTrace.x) return null;
    const albumId = gd.layout && gd.layout.uirevision;
    const ratings = barTrace.y.slice();
    const ignored = (barTrace.customdata||[]).map(cd=> !!(cd[1]));
    return {div: gd, album_id: albumId, ratings, ignored};
  }

  function handlePos(e, info, first){
    const xy = relToData(info.div, e.clientX, e.clientY);
    if(!xy) return;
    const xIndex = xy.xIndex;
    if(xIndex < 0 || xIndex >= info.ratings.length) return;
    if(info.ignored[xIndex]) return;
    const y = snap(xy.yVal);
    info.ratings[xIndex] = y;
    liveUpdate(info);
  }

  function relToData(gd, clientX, clientY){
    try {
      const full = gd._fullLayout;
      const plotBB = gd.querySelector('.plot');
      if(!plotBB) return null;
      const bb = plotBB.getBoundingClientRect();
      const px = clientX - bb.left;
      const py = clientY - bb.top;
      if(px<0||py<0||px>bb.width||py>bb.height) return null;
      const xFrac = px / bb.width; // 0..1
      const yFrac = 1 - (py / bb.height); // invert y
      // x axis ticks are integer positions starting at 1
      const nBars = gd.data[0].x.length;
      const xIndex = Math.floor(xFrac * nBars);
      const yVal = yFrac * 10; // since fixed 0..10
      return {xIndex, yVal};
    } catch(err){ log('relToData error', err); return null; }
  }

  function liveUpdate(info){
    const gd = info.div;
    const newY = info.ratings.map((v,i)=> info.ignored[i] || v==null ? 0 : v);
    Plotly.restyle(gd, {y:[newY]}, [0]);
  }

  let persistTimer = null;
  function persist(info){
    if(persistTimer) clearTimeout(persistTimer);
    persistTimer = setTimeout(()=>{
      log('Persist', info.album_id, info.ratings);
      fetch('/api/rate', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({album_id: info.album_id, ratings: info.ratings, ignored: info.ignored})
      }).catch(err=>log('Persist error', err));
    }, 150);
  }

  function scan(){ findGraphDivs().forEach(init); }
  const obs = new MutationObserver(scan); obs.observe(document.body, {subtree:true, childList:true});
  window.addEventListener('load', scan); scan();
})();
