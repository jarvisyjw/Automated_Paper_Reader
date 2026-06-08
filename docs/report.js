const params = new URLSearchParams(window.location.search);
const reportFile = params.get('file');
const titleEl = document.getElementById('report-title');
const metaEl = document.getElementById('report-meta');
const contentEl = document.getElementById('report-content');

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function inlineMarkdown(text) {
    let html = escapeHtml(text);
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    html = html.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    return html;
}

function flushList(listItems, html) {
    if (!listItems.length) return;
    html.push('<ul>');
    listItems.forEach((item) => html.push(`<li>${inlineMarkdown(item)}</li>`));
    html.push('</ul>');
    listItems.length = 0;
}

function renderMarkdown(markdown) {
    const html = [];
    const listItems = [];
    const lines = markdown.replace(/\r\n/g, '\n').split('\n');
    let paragraph = [];

    function flushParagraph() {
        if (!paragraph.length) return;
        html.push(`<p>${inlineMarkdown(paragraph.join(' '))}</p>`);
        paragraph = [];
    }

    lines.forEach((line) => {
        const trimmed = line.trim();

        if (!trimmed) {
            flushParagraph();
            flushList(listItems, html);
            return;
        }

        const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
        if (heading) {
            flushParagraph();
            flushList(listItems, html);
            const level = Math.min(heading[1].length, 4);
            html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
            return;
        }

        const bullet = trimmed.match(/^[-*]\s+(.+)$/);
        if (bullet) {
            flushParagraph();
            listItems.push(bullet[1]);
            return;
        }

        flushList(listItems, html);
        paragraph.push(trimmed);
    });

    flushParagraph();
    flushList(listItems, html);
    return html.join('\n');
}

function reportTitleFromFile(file) {
    const name = file.split('/').pop().replace(/\.md$/i, '');
    return `AIPR Report - ${name}`;
}

(async () => {
    if (!reportFile || !/^reports\/[A-Za-z0-9._-]+\.md$/.test(reportFile)) {
        metaEl.textContent = 'Missing or invalid report file.';
        contentEl.innerHTML = '<p>Open a report from the report list.</p>';
        return;
    }

    titleEl.textContent = reportTitleFromFile(reportFile);
    metaEl.textContent = reportFile;

    try {
        const response = await fetch(reportFile);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const markdown = await response.text();
        contentEl.innerHTML = renderMarkdown(markdown);
        const firstHeading = contentEl.querySelector('h1');
        if (firstHeading) titleEl.textContent = firstHeading.textContent;
    } catch (error) {
        metaEl.textContent = 'Could not load report.';
        contentEl.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
    }
})();
