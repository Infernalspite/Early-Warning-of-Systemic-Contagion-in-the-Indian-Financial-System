// ================================================================
//  SYSTEMIC CONTAGION ENGINE — app.js v2.1
//  Fixed: ApexCharts datetime format, SVG sizing, absorption_ratio
// ================================================================

// ── Global State ────────────────────────────────────────────────
let dashboardData   = null;
let activeTabId     = 'tab-overview';

let criChart        = null;
let bankChart       = null;
let featureChart    = null;

let currentNetworkIndex  = 0;
let isNetworkPlaying     = false;
let networkPlayInterval  = null;

let priceMode = 'norm';   // 'norm' | 'raw'

// ── Boot ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {

    // Sidebar tab routing
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            switchTab(item.getAttribute('data-tab'));
        });
    });

    // Load data then init
    try {
        const res = await fetch('data/dashboard_data.json');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        dashboardData = await res.json();
        console.log('✅ Dashboard data loaded:', Object.keys(dashboardData));
        initAll();
    } catch (err) {
        console.error('❌ Data load failed:', err);
        document.body.innerHTML = `
            <div style="display:flex;height:100vh;align-items:center;justify-content:center;flex-direction:column;gap:16px;font-family:sans-serif;color:#9ca3af;background:#070a13;">
                <div style="font-size:36px;">⚠️</div>
                <h2 style="color:#f3f4f6;">Dashboard Data Not Found</h2>
                <p>Please run <code style="background:#1e293b;padding:4px 8px;border-radius:4px;color:#3b82f6;">python train_model.py</code> to generate the data.</p>
                <p style="font-size:12px;">Error: ${err.message}</p>
            </div>`;
    }
});

function initAll() {
    initOverviewTab();
    initNetworkTab();
    initBankTab();
    initModelTab();
    window.addEventListener('resize', onResize);
}

function onResize() {
    if (criChart)     criChart.resize();
    if (bankChart)    bankChart.resize();
    if (featureChart) featureChart.resize();
    if (activeTabId === 'tab-network') drawNetworkGraph();
}

// ── Tab Switch ───────────────────────────────────────────────────
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active-tab'));
    document.getElementById(tabId).classList.add('active-tab');
    activeTabId = tabId;

    if (tabId === 'tab-network') {
        // Give the DOM a frame to lay out before we read clientWidth/Height
        requestAnimationFrame(() => drawNetworkGraph());
    }
}

// ── Helper: build ApexCharts [timestamp, value] data ─────────────
function toDateSeries(values) {
    return values.map((v, i) => ({
        x: new Date(dashboardData.dates[i]).getTime(),
        y: v != null ? +v.toFixed(2) : null
    }));
}

// ================================================================
//  TAB 1 — CRI OVERVIEW
// ================================================================
function initOverviewTab() {
    const dates  = dashboardData.dates;
    const cri    = dashboardData.cri;
    const macro  = dashboardData.macro;
    const lastI  = dates.length - 1;

    // ── Key Metrics ──────────────────────────────────────────────
    const latestCRI = cri[lastI] ?? 0;
    const prevCRI   = cri[Math.max(0, lastI - 5)] ?? latestCRI;
    const delta     = latestCRI - prevCRI;

    document.getElementById('metric-cri').textContent = latestCRI.toFixed(1);

    const deltaEl = document.getElementById('metric-cri-delta');
    if (delta >= 0) {
        deltaEl.className = 'metric-subtext up';
        deltaEl.innerHTML = `<i class="fa-solid fa-caret-up"></i> +${delta.toFixed(1)} vs 5d ago`;
    } else {
        deltaEl.className = 'metric-subtext down';
        deltaEl.innerHTML = `<i class="fa-solid fa-caret-down"></i> ${delta.toFixed(1)} vs 5d ago`;
    }

    // Risk pulse indicator
    const riskPulse = document.getElementById('risk-pulse');
    const riskText  = document.getElementById('risk-level-text');
    riskPulse.className = 'pulse-indicator';
    if (latestCRI >= 60)      { riskPulse.classList.add('pulse-red');    riskText.textContent = '🔴  HIGH RISK'; }
    else if (latestCRI >= 30) { riskPulse.classList.add('pulse-yellow'); riskText.textContent = '🟡  MEDIUM RISK'; }
    else                      { riskPulse.classList.add('pulse-green');  riskText.textContent = '🟢  LOW RISK'; }

    // VIX
    const vix = macro.india_vix?.[lastI] ?? 0;
    document.getElementById('metric-vix').textContent = vix.toFixed(1);
    const vixSub = document.getElementById('metric-vix-subtext');
    if (vix >= 25)      vixSub.innerHTML = `<span class="color-red"><i class="fa-solid fa-circle-exclamation"></i> High Stress</span>`;
    else if (vix >= 18) vixSub.innerHTML = `<span class="color-yellow"><i class="fa-solid fa-triangle-exclamation"></i> Moderate Stress</span>`;
    else                vixSub.innerHTML = `<span class="color-green"><i class="fa-solid fa-circle-check"></i> Normal Volatility</span>`;

    // Repo Rate
    const repo = macro.repo_rate?.[lastI] ?? 0;
    document.getElementById('metric-repo').textContent = `${repo.toFixed(2)}%`;

    // INR/USD
    const inr = macro.inr_usd?.[lastI] ?? 0;
    document.getElementById('metric-inr').textContent = `₹${inr.toFixed(2)}`;

    // ── CRI Chart ────────────────────────────────────────────────
    criChart = new ApexCharts(document.getElementById('cri-timeline-chart'), {
        series: [{ name: 'Contagion Risk Index', data: toDateSeries(cri) }],
        chart: {
            type: 'area',
            height: 300,
            background: 'transparent',
            fontFamily: 'Plus Jakarta Sans',
            foreColor: '#9ca3af',
            toolbar: { show: false },
            animations: { enabled: true, easing: 'easeinout', speed: 700 },
            zoom: { enabled: true }
        },
        dataLabels: { enabled: false },
        stroke: { curve: 'smooth', width: 2, colors: ['#ef4444'] },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.40, opacityTo: 0.04,
                colorStops: [
                    { offset: 0,   color: '#ef4444', opacity: 0.40 },
                    { offset: 100, color: '#ef4444', opacity: 0.04 }
                ]
            }
        },
        grid: { borderColor: 'rgba(255,255,255,0.05)', strokeDashArray: 4 },
        xaxis: {
            type: 'datetime',
            axisBorder: { show: false },
            axisTicks: { show: false },
            labels: { datetimeUTC: false }
        },
        yaxis: { min: 0, max: 100, labels: { formatter: v => v.toFixed(0) } },
        tooltip: { theme: 'dark', x: { format: 'dd MMM yyyy' } },
        annotations: {
            xaxis: [
                { x: new Date('2018-09-21').getTime(), borderColor: '#f59e0b',
                  label: { borderColor: '#f59e0b', style: { color: '#fff', background: '#f59e0b' }, text: 'IL&FS' } },
                { x: new Date('2020-03-05').getTime(), borderColor: '#ef4444',
                  label: { borderColor: '#ef4444', style: { color: '#fff', background: '#ef4444' }, text: 'Yes Bank' } },
                { x: new Date('2020-03-23').getTime(), borderColor: '#ef4444',
                  label: { borderColor: '#ef4444', style: { color: '#fff', background: '#ef4444' }, text: 'COVID' } },
                { x: new Date('2023-01-24').getTime(), borderColor: '#3b82f6',
                  label: { borderColor: '#3b82f6', style: { color: '#fff', background: '#3b82f6' }, text: 'Adani' } }
            ]
        }
    });
    criChart.render();

    // Range buttons
    document.querySelectorAll('.chart-actions .btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.chart-actions .btn').forEach(b => b.classList.remove('active-range'));
            btn.classList.add('active-range');
            zoomRange(criChart, btn.dataset.range);
        });
    });

    // Crisis card clicks
    document.querySelectorAll('.crisis-card').forEach(card => {
        card.addEventListener('click', () => {
            const dateStr = card.dataset.date;
            const t = new Date(dateStr).getTime();
            criChart.zoomX(t - 30 * 864e5, t + 60 * 864e5);

            const idx = dashboardData.network_history.findIndex(n => n.date === dateStr);
            if (idx !== -1) {
                currentNetworkIndex = idx;
                document.querySelector('.nav-item[data-tab="tab-network"]').click();
            }
        });
    });
}

function zoomRange(chart, range) {
    const dates   = dashboardData.dates;
    const lastT   = new Date(dates[dates.length - 1]).getTime();
    const firstT  = new Date(dates[0]).getTime();
    const yr = 365 * 864e5;
    const startT  = range === '1y' ? lastT - yr
                  : range === '2y' ? lastT - 2 * yr
                  : range === '5y' ? lastT - 5 * yr
                  : firstT;
    chart.zoomX(startT, lastT);
}

// ================================================================
//  TAB 2 — CONTAGION NETWORK
// ================================================================
function initNetworkTab() {
    const n = dashboardData.network_history.length;
    const slider = document.getElementById('network-time-slider');
    slider.max   = n - 1;
    slider.value = 0;
    currentNetworkIndex = 0;

    slider.addEventListener('input', e => {
        currentNetworkIndex = +e.target.value;
        drawNetworkGraph();
    });

    document.getElementById('corr-threshold-slider').addEventListener('input', e => {
        document.getElementById('corr-threshold-value').textContent = (+e.target.value).toFixed(2);
        drawNetworkGraph();
    });

    document.getElementById('network-play-btn').addEventListener('click', () => {
        isNetworkPlaying ? pauseNetwork() : playNetwork();
    });

    document.getElementById('play-speed').addEventListener('change', () => {
        if (isNetworkPlaying) { pauseNetwork(); playNetwork(); }
    });
}

function playNetwork() {
    isNetworkPlaying = true;
    const btn = document.getElementById('network-play-btn');
    btn.innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
    btn.classList.replace('btn-primary', 'btn-outline');

    const speed = +document.getElementById('play-speed').value;
    networkPlayInterval = setInterval(() => {
        currentNetworkIndex = (currentNetworkIndex + 1) % dashboardData.network_history.length;
        document.getElementById('network-time-slider').value = currentNetworkIndex;
        drawNetworkGraph();
    }, speed);
}

function pauseNetwork() {
    isNetworkPlaying = false;
    const btn = document.getElementById('network-play-btn');
    btn.innerHTML = '<i class="fa-solid fa-play"></i> Play';
    btn.classList.replace('btn-outline', 'btn-primary');
    clearInterval(networkPlayInterval);
    networkPlayInterval = null;
}

function drawNetworkGraph() {
    if (activeTabId !== 'tab-network' || !dashboardData) return;

    const record = dashboardData.network_history[currentNetworkIndex];
    if (!record) return;

    // ── Date label ────────────────────────────────────────────────
    document.getElementById('network-current-date').textContent =
        new Date(record.date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });

    // ── CRI for this date ─────────────────────────────────────────
    const timelineIdx = dashboardData.dates.indexOf(record.date);
    const dateCRI = timelineIdx !== -1 ? (dashboardData.cri[timelineIdx] ?? 0) : 0;

    const badge = document.getElementById('network-risk-badge');
    badge.className = 'risk-label-badge';
    if (dateCRI >= 60)      { badge.classList.add('risk-high');   badge.textContent = `High Risk  (CRI: ${dateCRI.toFixed(1)})`; }
    else if (dateCRI >= 30) { badge.classList.add('risk-medium'); badge.textContent = `Medium Risk (CRI: ${dateCRI.toFixed(1)})`; }
    else                    { badge.classList.add('risk-low');    badge.textContent = `Low Risk  (CRI: ${dateCRI.toFixed(1)})`; }

    // ── SVG dimensions ────────────────────────────────────────────
    const wrapper = document.getElementById('contagion-network-svg-container');
    wrapper.innerHTML = '';
    const W = wrapper.clientWidth  || 700;
    const H = wrapper.clientHeight || 480;
    if (W < 10 || H < 10) return;   // not yet laid out

    const svg = d3.select(wrapper)
        .append('svg')
        .attr('width',   W)
        .attr('height',  H)
        .attr('viewBox', `0 0 ${W} ${H}`);

    // Glow filter
    const defs = svg.append('defs');
    const addGlow = (id, color, blur) => {
        const f = defs.append('filter').attr('id', id)
            .attr('x', '-30%').attr('y', '-30%').attr('width', '160%').attr('height', '160%');
        f.append('feGaussianBlur').attr('in', 'SourceGraphic').attr('stdDeviation', blur).attr('result', 'blur');
        const merge = f.append('feMerge');
        merge.append('feMergeNode').attr('in', 'blur');
        merge.append('feMergeNode').attr('in', 'SourceGraphic');
    };
    addGlow('glow-red',    '#ef4444', 5);
    addGlow('glow-blue',   '#3b82f6', 4);
    addGlow('glow-purple', '#8b5cf6', 4);

    // ── Node positions (stable circular arcs) ─────────────────────
    const PSU_BANKS = [
        'State Bank of India', 'Punjab National Bank', 'Bank of Baroda', 'Canara Bank',
        'Union Bank of India', 'Indian Bank', 'Indian Overseas Bank', 'Bank of Maharashtra'
    ];
    const allBanks    = dashboardData.bank_names;
    const psuBanks    = allBanks.filter(b => PSU_BANKS.includes(b));
    const privateBanks = allBanks.filter(b => !PSU_BANKS.includes(b));

    const cx1 = W * 0.28, cy1 = H * 0.50, r1 = Math.min(H * 0.36, 140);
    const cx2 = W * 0.72, cy2 = H * 0.50, r2 = Math.min(H * 0.36, 140);

    function arcPos(idx, total, cx, cy, r, startAngle, endAngle) {
        const t = total <= 1 ? 0.5 : idx / (total - 1);
        const a = startAngle + t * (endAngle - startAngle);
        return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
    }

    const nodes = allBanks.map(bank => {
        const isPSU = PSU_BANKS.includes(bank);
        let pos;
        if (isPSU) {
            const i = psuBanks.indexOf(bank);
            pos = arcPos(i, psuBanks.length, cx1, cy1, r1, Math.PI * 0.65, Math.PI * 1.35);
        } else {
            const i = privateBanks.indexOf(bank);
            pos = arcPos(i, privateBanks.length, cx2, cy2, r2, -Math.PI * 0.35, Math.PI * 0.35);
        }
        return { id: bank, isPSU, x: pos.x, y: pos.y, r: 9 };
    });

    // ── Links ─────────────────────────────────────────────────────
    const threshold = +document.getElementById('corr-threshold-slider').value;
    const rawLinks  = (record.links || []).filter(l => l.value >= threshold);
    const links = rawLinks.map(l => ({
        source: nodes.find(n => n.id === l.source),
        target: nodes.find(n => n.id === l.target),
        value: l.value
    })).filter(l => l.source && l.target);

    // Sidebar stats
    document.getElementById('net-stat-links').textContent   = links.length;
    const density = (links.length / ((allBanks.length * (allBanks.length - 1)) / 2)) * 100;
    document.getElementById('net-stat-density').textContent = `${density.toFixed(1)}%`;
    document.getElementById('net-stat-pca').textContent     = 'N/A';  // absorption_ratio not in JSON

    // ── Draw links ────────────────────────────────────────────────
    const linkEls = svg.append('g').attr('class', 'links')
        .selectAll('line').data(links).enter().append('line')
        .attr('class', 'link-line')
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
        .style('stroke', d => d.value >= 0.65 ? '#ef4444' : d.value >= 0.50 ? '#f59e0b' : '#3b82f6')
        .style('stroke-opacity', d => 0.15 + (d.value - 0.40) * 1.2)
        .style('stroke-width',   d => (d.value - 0.40) * 9)
        .style('filter', d => d.value >= 0.65 ? 'url(#glow-red)' : 'none');

    // ── Draw nodes ────────────────────────────────────────────────
    const nodeGs = svg.append('g').attr('class', 'nodes')
        .selectAll('g').data(nodes).enter().append('g')
        .on('mouseover', function(_, d) {
            d3.select(this).select('circle')
                .attr('r', d.r + 5)
                .style('filter', 'url(#glow-red)');
            linkEls
                .style('stroke-opacity', l => (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.04)
                .style('stroke',         l => (l.source.id === d.id || l.target.id === d.id) ? '#ef4444' : 'rgba(255,255,255,0.08)');
        })
        .on('mouseout', function(_, d) {
            d3.select(this).select('circle')
                .attr('r', d.r)
                .style('filter', d.isPSU && dateCRI >= 50 ? 'url(#glow-purple)' : 'url(#glow-blue)');
            linkEls
                .style('stroke-opacity', l => 0.15 + (l.value - 0.40) * 1.2)
                .style('stroke', l => l.value >= 0.65 ? '#ef4444' : l.value >= 0.50 ? '#f59e0b' : '#3b82f6');
        });

    nodeGs.append('circle')
        .attr('cx', d => d.x).attr('cy', d => d.y).attr('r', d => d.r)
        .attr('class', 'node-circle')
        .style('fill',   d => d.isPSU ? '#8b5cf6' : '#3b82f6')
        .style('filter', d => d.isPSU && dateCRI >= 50 ? 'url(#glow-purple)' : 'url(#glow-blue)');

    // Shortened labels
    const ABBR = {
        'State Bank of India': 'SBI', 'Punjab National Bank': 'PNB',
        'Bank of Baroda': 'BOB', 'Canara Bank': 'CANARA',
        'Union Bank of India': 'UNION', 'Indian Bank': 'IND BNK',
        'Indian Overseas Bank': 'IOB', 'Bank of Maharashtra': 'MAHAR',
        'HDFC Bank': 'HDFC', 'ICICI Bank': 'ICICI', 'Axis Bank': 'AXIS',
        'Kotak Mahindra Bank': 'KOTAK', 'IndusInd Bank': 'INDUS',
        'Yes Bank': 'YES', 'Bandhan Bank': 'BANDHAN', 'IDFC First Bank': 'IDFC',
        'Federal Bank': 'FEDERAL', 'AU Small Finance Bank': 'AU SFB',
        'RBL Bank': 'RBL', 'CSB Bank': 'CSB'
    };
    nodeGs.append('text')
        .attr('x', d => d.x)
        .attr('y', d => d.isPSU ? d.y - d.r - 4 : d.y + d.r + 12)
        .attr('text-anchor', 'middle')
        .attr('class', 'node-label')
        .text(d => ABBR[d.id] || d.id.split(' ')[0]);

    // Section divider labels
    svg.append('text').attr('x', cx1).attr('y', 18)
        .attr('text-anchor', 'middle')
        .style('fill', '#8b5cf6').style('font-size', '10px').style('font-weight', '700')
        .style('font-family', 'Plus Jakarta Sans').style('letter-spacing', '1px')
        .text('◼  PSU BANKS');
    svg.append('text').attr('x', cx2).attr('y', 18)
        .attr('text-anchor', 'middle')
        .style('fill', '#3b82f6').style('font-size', '10px').style('font-weight', '700')
        .style('font-family', 'Plus Jakarta Sans').style('letter-spacing', '1px')
        .text('◼  PRIVATE BANKS');
}

// ================================================================
//  TAB 3 — BANK COMPARISON
// ================================================================
function initBankTab() {
    const banks   = dashboardData.bank_names;
    const defaults = ['State Bank of India', 'HDFC Bank', 'Yes Bank', 'ICICI Bank'];
    const listEl  = document.getElementById('bank-selectors-container');
    listEl.innerHTML = '';

    banks.forEach(bank => {
        const checked = defaults.includes(bank);
        const lbl = document.createElement('label');
        lbl.className = `checkbox-item${checked ? ' checked' : ''}`;
        lbl.innerHTML = `<input type="checkbox" value="${bank}"${checked ? ' checked' : ''}><span>${bank}</span>`;
        lbl.querySelector('input').addEventListener('change', e => {
            lbl.classList.toggle('checked', e.target.checked);
            bankChart?.updateSeries(buildBankSeries());
        });
        listEl.appendChild(lbl);
    });

    document.getElementById('select-all-banks').addEventListener('click', () => {
        document.querySelectorAll('.checkbox-item').forEach(i => { i.classList.add('checked'); i.querySelector('input').checked = true; });
        bankChart?.updateSeries(buildBankSeries());
    });
    document.getElementById('deselect-all-banks').addEventListener('click', () => {
        document.querySelectorAll('.checkbox-item').forEach(i => { i.classList.remove('checked'); i.querySelector('input').checked = false; });
        bankChart?.updateSeries([]);
    });

    // Normalized / Raw toggle
    const btnN = document.getElementById('btn-price-norm');
    const btnR = document.getElementById('btn-price-raw');
    btnN.addEventListener('click', () => { priceMode = 'norm'; btnN.classList.add('active'); btnR.classList.remove('active'); bankChart?.updateSeries(buildBankSeries()); });
    btnR.addEventListener('click', () => { priceMode = 'raw';  btnR.classList.add('active'); btnN.classList.remove('active'); bankChart?.updateSeries(buildBankSeries()); });

    bankChart = new ApexCharts(document.getElementById('bank-prices-comparison-chart'), {
        series: buildBankSeries(),
        chart: {
            type: 'line',
            height: '100%',
            background: 'transparent',
            fontFamily: 'Plus Jakarta Sans',
            foreColor: '#9ca3af',
            toolbar: { show: false },
            animations: { enabled: false }   // disable for performance with many series
        },
        stroke: { width: 1.5, curve: 'straight' },
        grid: { borderColor: 'rgba(255,255,255,0.05)', strokeDashArray: 4 },
        xaxis: {
            type: 'datetime',
            axisBorder: { show: false },
            axisTicks: { show: false },
            labels: { datetimeUTC: false }
        },
        yaxis: { title: { text: 'Normalised (Base = 100)' } },
        tooltip: { theme: 'dark', x: { format: 'dd MMM yyyy' } },
        legend: { position: 'top', horizontalAlign: 'left', fontSize: '11px' }
    });
    bankChart.render();
}

function buildBankSeries() {
    const checked = document.querySelectorAll('#bank-selectors-container input:checked');
    return [...checked].map(cb => {
        const bp = dashboardData.bank_prices[cb.value];
        if (!bp) return null;
        return { name: cb.value, data: toDateSeries(priceMode === 'norm' ? bp.norm : bp.raw) };
    }).filter(Boolean);
}

// ================================================================
//  TAB 4 — MODEL INSIGHTS
// ================================================================
function initModelTab() {
    const metrics = dashboardData.model_metrics;
    document.getElementById('spec-accuracy').textContent     = `${(metrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById('spec-features-count').textContent = `${metrics.features_used.length} Features`;

    const top10 = [...dashboardData.feature_importances]
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 10);

    const prettyLabel = s => s
        .replace(/_/g, ' ')
        .replace(/\b(\w)/g, c => c.toUpperCase())
        .replace('5D', '(5-day)')
        .replace('10D', '(10-day)')
        .replace('30D', '(30-day)');

    featureChart = new ApexCharts(document.getElementById('model-feature-importance-chart'), {
        series: [{ name: 'Importance', data: top10.map(f => +f.importance.toFixed(4)) }],
        chart: {
            type: 'bar',
            height: '100%',
            background: 'transparent',
            fontFamily: 'Plus Jakarta Sans',
            foreColor: '#9ca3af',
            toolbar: { show: false }
        },
        plotOptions: { bar: { horizontal: true, borderRadius: 4, barHeight: '65%' } },
        colors: ['#3b82f6'],
        grid: { borderColor: 'rgba(255,255,255,0.05)', strokeDashArray: 4 },
        xaxis: {
            categories: top10.map(f => prettyLabel(f.feature)),
            axisBorder: { show: false },
            axisTicks: { show: false },
            labels: { style: { fontSize: '10px' } }
        },
        tooltip: { theme: 'dark' }
    });
    featureChart.render();
}
