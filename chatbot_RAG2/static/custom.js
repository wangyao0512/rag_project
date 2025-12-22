function() {
  console.log('Citation system JS loaded (NEW span payload only)');

  // remove old listener (capture=true)
  const oldListener = window.citationClickListener;
  if (oldListener) document.removeEventListener('click', oldListener, true);

  const b64ToUtf8 = (b64) => {
    // 优先 TextDecoder；失败则回退老办法（避免某些环境偶发解码空）
    try {
      const bin = atob((b64 || '').trim());
      const bytes = Uint8Array.from(bin, c => c.charCodeAt(0));
      return new TextDecoder('utf-8').decode(bytes);
    } catch (e) {
      const s = atob((b64 || '').trim());
      // return decodeURIComponent(escape(s));
      return decodeURIComponent(escape(s));
    }
  };

  const parseDisplayNum = (a) => {
    const m = (a.textContent || '').match(/\[(\d+)\]/);
    return m ? m[1] : null;
  };

  const getPayloadB64 = (a) => {
    // href="#cite-xxx"
    const href = a.getAttribute('href') || '';
    if (href.startsWith('#')) {
      const id = href.slice(1);
      const el = document.getElementById(id);
      if (el && el.tagName === 'SPAN' && el.classList.contains('citation-payload')) {
        const b64 = (el.textContent || '').trim();
        if (b64) return { b64, id, via: 'href->span#id' };
      }
    }

    // 兜底：紧邻 span
    const next = a.nextElementSibling;
    if (next && next.tagName === 'SPAN' && next.classList.contains('citation-payload')) {
      const b64 = (next.textContent || '').trim();
      if (b64) return { b64, id: next.id || null, via: 'nextElementSibling span' };
    }

    return null;
  };

  const closeAll = () => document.querySelectorAll('.citation-tooltip').forEach(t => t.remove());

  const clickHandler = (e) => {
    const a = e.target.closest('a.citation-link');

    if (a) {
      e.preventDefault();
      // ✅ 防止 Gradio/其它全局 click handler 把 tooltip 秒删
      e.stopImmediatePropagation();

      closeAll();

      const displayNum = parseDisplayNum(a);
      const payload = getPayloadB64(a);
      if (!payload) {
        console.warn('No citation payload span found (likely sanitized/missing):', {
          displayNum,
          href: a.getAttribute('href'),
          html: a.outerHTML
        });
        return;
      }

      let obj = null;
      try {
        obj = JSON.parse(b64ToUtf8(payload.b64));
      } catch (err) {
        console.error('Decode/JSON parse failed:', err, payload);
        return;
      }

      const content = (obj && obj.content) ? String(obj.content) : '';
      if (!content) {
        console.warn('Citation content empty:', obj);
        return;
      }

      const tooltip = document.createElement('div');
      tooltip.className = 'citation-tooltip';
      tooltip.style.position = 'absolute';
      tooltip.style.zIndex = '999999'; // ✅ 很关键：避免被遮住

      const closeBtn = document.createElement('span');
      closeBtn.className = 'citation-tooltip-close';
      closeBtn.innerHTML = '×';
      closeBtn.onclick = (ev) => { ev.stopImmediatePropagation(); tooltip.remove(); };
      tooltip.appendChild(closeBtn);

      const head = document.createElement('div');
      head.className = 'citation-source-info';
      let headText = obj.source || '未知来源';
      if (obj.year && !headText.includes('(' + obj.year + ')')) headText += ` (${obj.year})`;
      if (displayNum) headText = `[${displayNum}] ` + headText;
      head.textContent = headText;
      tooltip.appendChild(head);

      const body = document.createElement('div');
      body.className = 'citation-content';
      // body.textContent = content;
      // const escapedContent = content.replace(/~/g, '&#126;');
      // body.innerHTML = escapedContent;
      body.textContent = content;
      tooltip.appendChild(body);

      document.body.appendChild(tooltip);

      // position
      const r = a.getBoundingClientRect();
      const tr = tooltip.getBoundingClientRect();
      const M = 10;

      const viewportH = window.innerHeight;
      const viewportW = window.innerWidth;
      const scrollY = window.scrollY;
      const scrollX = window.scrollX;

      const spaceBelow = viewportH - r.bottom;
      const spaceAbove = r.top;

      let top;
      if (spaceBelow >= tr.height + M) top = r.bottom + scrollY + 6;
      else if (spaceAbove >= tr.height + M) top = r.top + scrollY - tr.height - 6;
      else top = scrollY + M;

      let left = r.left + scrollX;
      left = Math.max(M, Math.min(left, scrollX + viewportW - tr.width - M));

      tooltip.style.top = `${top}px`;
      tooltip.style.left = `${left}px`;
      return;
    }

    // outside click closes
    if (!e.target.closest('.citation-tooltip')) closeAll();
  };

  window.citationClickListener = clickHandler;
  document.addEventListener('click', clickHandler, true);
}