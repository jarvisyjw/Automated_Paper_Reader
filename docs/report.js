const params = new URLSearchParams(window.location.search);
const reportFile = params.get('file');
let currentLang = params.get('lang') === 'zh' ? 'zh' : 'en';
const titleEl = document.getElementById('report-title');
const metaEl = document.getElementById('report-meta');
const contentEl = document.getElementById('report-content');
let pageTitle = '';

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
    return `${name}_report`;
}

function localizedReportFile(file, lang) {
    const name = file.split('/').pop();
    return lang === 'zh' ? `reports/zh/${name}` : `reports/${name}`;
}

function setUrlLang(lang) {
    const nextParams = new URLSearchParams(window.location.search);
    nextParams.set('lang', lang);
    window.history.replaceState({}, '', `${window.location.pathname}?${nextParams.toString()}`);
}

function insertLanguageSwitch() {
    const existing = contentEl.querySelector('.language-switch');
    if (existing) existing.remove();

    const switchEl = document.createElement('div');
    switchEl.className = 'language-switch';
    switchEl.innerHTML = `
        <span>Language</span>
        <button type="button" data-lang="en">EN</button>
        <button type="button" data-lang="zh">中文</button>
    `;
    switchEl.querySelectorAll('button').forEach((button) => {
        const lang = button.getAttribute('data-lang');
        if (lang === currentLang) button.classList.add('active');
        button.addEventListener('click', async () => {
            if (lang === currentLang) return;
            currentLang = lang;
            setUrlLang(lang);
            await loadReport();
        });
    });

    const firstHeading = contentEl.querySelector('h1');
    if (firstHeading) {
        firstHeading.insertAdjacentElement('afterend', switchEl);
    } else {
        contentEl.prepend(switchEl);
    }
}

async function loadReport() {
    if (!reportFile || !/^reports\/[A-Za-z0-9._-]+\.md$/.test(reportFile)) {
        metaEl.textContent = 'Missing or invalid report file.';
        contentEl.innerHTML = '<p>Open a report from the report list.</p>';
        return;
    }

    pageTitle = reportTitleFromFile(reportFile);
    titleEl.textContent = pageTitle;
    const targetFile = localizedReportFile(reportFile, currentLang);
    metaEl.textContent = currentLang === 'zh' ? `${targetFile} · 中文` : `${targetFile} · English`;

    try {
        const response = await fetch(targetFile);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const markdown = await response.text();
        contentEl.innerHTML = renderMarkdown(markdown);
        const firstHeading = contentEl.querySelector('h1');
        if (firstHeading && /^Daily Paper Report( Template)?$/i.test(firstHeading.textContent.trim())) {
            firstHeading.textContent = pageTitle;
        }
        insertLanguageSwitch();
    } catch (error) {
        metaEl.textContent = 'Could not load report.';
        contentEl.innerHTML = `
            <h1>${escapeHtml(pageTitle || 'Daily Paper Report')}</h1>
            <p>${escapeHtml(error.message)}</p>
        `;
        insertLanguageSwitch();
    }
}

loadReport();
