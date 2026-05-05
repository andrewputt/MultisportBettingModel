/**
 * server.js
 * ─────────────────────────────────────────────────────────────────────────────
 * EPL Predictions — Express API server.
 *
 * Provides two data streams to the frontend:
 *
 *   Stream 1 – Kalshi (KXEPLGAME series)
 *     GET /api/kalshi-epl
 *     Fetches all open EPL match markets, groups them into home/away/draw
 *     objects with yes_bid / yes_ask prices, and returns a ranked match list.
 *
 *   Stream 2 – API-Football (v3.football.api-sports.io)
 *     GET /api/h2h?home=<team>&away=<team>
 *     Resolves team names → real numeric IDs, fetches the last 20 H2H
 *     fixtures across all competitions (filtered to 3 years), computes
 *     win/draw/loss percentages and average goals server-side using real
 *     team IDs (no fragile string matching), and returns ready-to-render data.
 *
 * Auth
 *   Kalshi  : pass RSA key as  X-Kalshi-Key: <keyId>::<PEM>  header
 *             (public market data works without auth — signing is optional)
 *   Football: pass API key as  X-Football-Key: <key>  header
 *
 * Usage
 *   node server.js            # default port 3000
 *   PORT=8080 node server.js  # custom port
 *
 * Install
 *   npm install               # see package.json
 */

'use strict';

const express = require('express');
const fetch   = require('node-fetch');
const path    = require('path');
const crypto  = require('crypto');

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, '..', '..', 'frontend')));

// ── API base URLs ─────────────────────────────────────────────────────────────
const KALSHI_BASE   = 'https://api.elections.kalshi.com/trade-api/v2';
const FOOTBALL_BASE = 'https://v3.football.api-sports.io';


// ═════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═════════════════════════════════════════════════════════════════════════════

/**
 * kalshiGet
 * Authenticated GET against the Kalshi Trade API.
 * If the caller passes a key in the format  <keyId>::<PEM>  the request is
 * signed with RSA-PSS (sha256).  Without a key the request is sent unsigned
 * — public market data endpoints work either way.
 *
 * @param {string} urlPath  - Path + query string, e.g. '/markets?series_ticker=KXEPLGAME'
 * @param {string} kalshiKey - Optional '<keyId>::<PEM>' string from the header
 * @returns {Promise<object>}
 */
async function kalshiGet(urlPath, kalshiKey) {
  const headers = { 'Content-Type': 'application/json' };

  if (kalshiKey && kalshiKey.includes('::')) {
    try {
      const sep      = kalshiKey.indexOf('::');
      const keyId    = kalshiKey.slice(0, sep).trim();
      const privPem  = kalshiKey.slice(sep + 2).trim();
      const ts       = String(Date.now());
      const pathOnly = urlPath.split('?')[0];
      const msg      = ts + 'GET' + pathOnly;

      const privateKey = crypto.createPrivateKey(privPem);
      const sig = crypto.sign('sha256', Buffer.from(msg), {
        key:        privateKey,
        padding:    crypto.constants.RSA_PKCS1_PSS_PADDING,
        saltLength: crypto.constants.RSA_PSS_SALTLEN_DIGEST,
      });

      headers['KALSHI-ACCESS-KEY']       = keyId;
      headers['KALSHI-ACCESS-SIGNATURE'] = sig.toString('base64');
      headers['KALSHI-ACCESS-TIMESTAMP'] = ts;
    } catch (e) {
      console.warn('Kalshi signing failed, using public access:', e.message);
    }
  }

  const res = await fetch(`${KALSHI_BASE}${urlPath}`, { headers });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Kalshi ${res.status}: ${body || res.statusText}`);
  }
  return res.json();
}


/**
 * footballGet
 * GET against the API-Football v3 endpoint.
 * Returns the `response` array from the API-Football JSON envelope.
 *
 * @param {string} endpoint - e.g. '/teams?search=Liverpool'
 * @param {string} apiKey
 * @returns {Promise<Array>}
 */
async function footballGet(endpoint, apiKey) {
  const res = await fetch(`${FOOTBALL_BASE}${endpoint}`, {
    headers: { 'x-apisports-key': apiKey },
  });
  if (!res.ok) throw new Error(`API-Football ${res.status}: ${res.statusText}`);

  const json = await res.json();
  if (json.errors && Object.keys(json.errors).length)
    throw new Error(Object.values(json.errors).join(', '));

  return json.response || [];
}


// ═════════════════════════════════════════════════════════════════════════════
// KALSHI — market grouping
// ═════════════════════════════════════════════════════════════════════════════

/**
 * fetchAllEplMarkets
 * Pages through all open KXEPLGAME markets (up to 2,000 total).
 * Returns a flat array of raw Kalshi market objects.
 *
 * @param {string} kalshiKey
 * @returns {Promise<Array>}
 */
async function fetchAllEplMarkets(kalshiKey) {
  const markets = [];
  let cursor = '', page = 0;

  do {
    const qs   = `series_ticker=KXEPLGAME&status=open&limit=200${cursor ? '&cursor=' + cursor : ''}`;
    const data = await kalshiGet(`/markets?${qs}`, kalshiKey);
    const batch = data.markets || [];
    markets.push(...batch);
    cursor = data.cursor || '';
    page++;
    if (!batch.length) break;
  } while (cursor && page < 10);

  return markets;
}


/**
 * groupIntoMatches
 * Converts a flat list of Kalshi market objects into structured match objects
 * with separate home / away / draw outcome slots.
 * Matches require at least two outcomes (home + away) to be included.
 *
 * @param {Array} markets - Raw Kalshi market array
 * @returns {Array<{event_ticker, home, away, close_time, home_odds, away_odds, draw_odds}>}
 */
function groupIntoMatches(markets) {
  // Group by event_ticker (one event = one match)
  const byEvent = {};
  for (const m of markets) {
    const et = m.event_ticker || m.ticker;
    if (!byEvent[et]) byEvent[et] = [];
    byEvent[et].push(m);
  }

  const matches = [];
  for (const [eventTicker, group] of Object.entries(byEvent)) {
    if (group.length < 2) continue;

    let home = null, away = null, draw = null;
    for (const m of group) {
      const label = (m.yes_sub_title || m.subtitle || '').toLowerCase().trim();
      const odds  = {
        ticker:     m.ticker,
        yes_bid:    m.yes_bid    ?? 0,
        yes_ask:    m.yes_ask    ?? 0,
        no_bid:     m.no_bid     ?? 0,
        last_price: m.last_price ?? 0,
        volume:     m.volume     ?? 0,
        label:      m.yes_sub_title || m.subtitle || m.ticker,
      };

      if (label === 'tie' || label === 'draw') draw = odds;
      else if (!home) home = odds;
      else if (!away) away = odds;
    }
    if (!home || !away) continue;

    // Extract team names from the event title ("Arsenal vs. Chelsea")
    const titleMatch = (group[0].title || '').match(/^(.+?)\s+vs\.?\s+(.+?)(?:\s*[?].*)?$/i);
    const homeName   = titleMatch ? titleMatch[1].trim() : home.label;
    const awayName   = titleMatch ? titleMatch[2].trim() : away.label;

    matches.push({
      event_ticker: eventTicker,
      home:         homeName,
      away:         awayName,
      close_time:   group[0].close_time || null,
      home_odds:    home,
      away_odds:    away,
      draw_odds:    draw,
    });
  }
  return matches;
}


// ═════════════════════════════════════════════════════════════════════════════
// API-FOOTBALL — H2H computation
// ═════════════════════════════════════════════════════════════════════════════

/**
 * computeH2HStats
 * Aggregates historical fixtures into win/draw/loss counts and average goals.
 * All comparisons use numeric team IDs — no string matching.
 *
 * Results are always from the perspective of the team passed as homeTeamId
 * (the "home" team in the matchup, not necessarily the venue home team).
 *
 * @param {Array}  fixtures   - API-Football fixture objects
 * @param {number} homeTeamId - Numeric ID of the team treated as "home"
 * @returns {object}
 */
function computeH2HStats(fixtures, homeTeamId) {
  let homeW = 0, awayW = 0, draws = 0, hG = 0, aG = 0;

  for (const f of fixtures) {
    const homeId = f.teams?.home?.id;
    const hg     = f.score?.fulltime?.home ?? null;
    const ag     = f.score?.fulltime?.away ?? null;

    // Skip fixtures with no score (postponed, cancelled, etc.)
    if (hg === null || ag === null) continue;

    const ourTeamIsHome = homeId === homeTeamId;
    const ourG  = ourTeamIsHome ? hg : ag;
    const oppG  = ourTeamIsHome ? ag : hg;

    hG += ourG;
    aG += oppG;

    if (ourG === oppG)      draws++;
    else if (ourG > oppG)   homeW++;
    else                    awayW++;
  }

  const n = fixtures.filter(
    f => f.score?.fulltime?.home !== null && f.score?.fulltime?.away !== null
  ).length || 1;

  return {
    homeW,
    awayW,
    draws,
    total:       fixtures.length,
    scored:      fixtures.filter(f => f.score?.fulltime?.home !== null).length,
    homeWinPct:  homeW / n,
    drawPct:     draws / n,
    awayWinPct:  awayW / n,
    avgHG:       n > 0 ? (hG / n).toFixed(2) : null,
    avgAG:       n > 0 ? (aG / n).toFixed(2) : null,
  };
}


/**
 * formatFixtures
 * Shapes raw API-Football fixture objects into the flat format the frontend
 * expects.  Result is always from the perspective of homeTeamId.
 *
 * @param {Array}  fixtures
 * @param {number} homeTeamId
 * @returns {Array}
 */
function formatFixtures(fixtures, homeTeamId) {
  return fixtures.map(f => {
    const homeId        = f.teams?.home?.id;
    const ourTeamIsHome = homeId === homeTeamId;
    const hg = f.score?.fulltime?.home;
    const ag = f.score?.fulltime?.away;
    const ourG = ourTeamIsHome ? hg : ag;
    const oppG = ourTeamIsHome ? ag : hg;

    let result = null;
    if (hg != null && ag != null) {
      if (ourG === oppG)    result = 'draw';
      else if (ourG > oppG) result = 'home_win';
      else                  result = 'away_win';
    }

    return {
      date:         f.fixture.date,
      competition:  f.league?.name || '',
      homeTeam:     f.teams?.home?.name || '',
      awayTeam:     f.teams?.away?.name || '',
      scoreHome:    ourG ?? null,
      scoreAway:    oppG ?? null,
      result,           // from perspective of homeTeamId
      ourTeamIsHome,
    };
  });
}


// ═════════════════════════════════════════════════════════════════════════════
// ROUTES
// ═════════════════════════════════════════════════════════════════════════════

/**
 * GET /api/kalshi-epl
 * Returns all open KXEPLGAME markets grouped into match objects.
 *
 * Headers
 *   X-Kalshi-Key: <keyId>::<PEM>   (optional — public data works without)
 *
 * Response
 *   { matches: [...], total_markets: number }
 */
app.get('/api/kalshi-epl', async (req, res) => {
  const kalshiKey = req.headers['x-kalshi-key'] || '';
  try {
    const allMarkets = await fetchAllEplMarkets(kalshiKey);
    console.log(`Kalshi: ${allMarkets.length} KXEPLGAME markets`);

    if (allMarkets[0]) {
      const s = allMarkets[0];
      console.log('Sample:', s.ticker, '|', s.title, '|', s.yes_sub_title, '| bid:', s.yes_bid);
    }

    const matches = groupIntoMatches(allMarkets);
    console.log(`Grouped: ${matches.length} matches`);

    res.json({ matches, total_markets: allMarkets.length });
  } catch (err) {
    console.error('Kalshi error:', err.message);
    res.status(500).json({ error: err.message, matches: [], total_markets: 0 });
  }
});


/**
 * GET /api/h2h?home=<team>&away=<team>
 * Resolves team names to real API-Football IDs, fetches last 20 H2H fixtures
 * (filtered to 3 years), computes stats server-side, returns structured data.
 *
 * Headers
 *   X-Football-Key: <key>   (required)
 *
 * Query params
 *   home  - Team name string, e.g. "Liverpool"
 *   away  - Team name string, e.g. "Arsenal"
 *
 * Response
 *   { stats, fixtures, homeId, awayId, homeTeamName, awayTeamName }
 */
app.get('/api/h2h', async (req, res) => {
  const apiKey = req.headers['x-football-key'];
  const { home, away } = req.query;

  if (!apiKey) return res.status(400).json({ error: 'Missing X-Football-Key header' });

  try {
    // Resolve team names → numeric IDs in parallel
    const [homeTeams, awayTeams] = await Promise.all([
      footballGet(`/teams?search=${encodeURIComponent(home)}`, apiKey),
      footballGet(`/teams?search=${encodeURIComponent(away)}`, apiKey),
    ]);

    const homeTeam = homeTeams[0];
    const awayTeam = awayTeams[0];
    const hId = homeTeam?.team?.id;
    const aId = awayTeam?.team?.id;

    console.log(`H2H lookup: "${home}" → id ${hId} (${homeTeam?.team?.name})`);
    console.log(`H2H lookup: "${away}" → id ${aId} (${awayTeam?.team?.name})`);

    if (!hId || !aId) {
      console.warn(`Could not find team IDs for: ${home} (${hId}) vs ${away} (${aId})`);
      return res.json({ stats: null, fixtures: [], homeId: null, awayId: null });
    }

    // Fetch last 20 H2H fixtures across all competitions
    const rawFixtures = await footballGet(
      `/fixtures/headtohead?h2h=${hId}-${aId}&last=20`,
      apiKey
    );

    // Filter to last 3 years only
    const cutoff  = new Date();
    cutoff.setFullYear(cutoff.getFullYear() - 3);
    const fixtures = rawFixtures.filter(f => new Date(f.fixture.date) >= cutoff);

    console.log(`H2H: ${rawFixtures.length} total → ${fixtures.length} in last 3 years`);

    // Compute stats and format fixtures using real team IDs (no string matching)
    const stats             = computeH2HStats(fixtures, hId);
    const formattedFixtures = formatFixtures(fixtures, hId);

    res.json({
      stats,
      fixtures:     formattedFixtures,
      homeId:       hId,
      awayId:       aId,
      homeTeamName: homeTeam?.team?.name,
      awayTeamName: awayTeam?.team?.name,
    });
  } catch (err) {
    console.error('H2H error:', err.message);
    res.json({ stats: null, fixtures: [], homeId: null, awayId: null });
  }
});


/**
 * GET /api/kalshi-debug
 * Returns the first 5 raw KXEPLGAME market objects for inspection.
 * Useful for verifying field names / structure during development.
 */
app.get('/api/kalshi-debug', async (req, res) => {
  const kalshiKey = req.headers['x-kalshi-key'] || '';
  try {
    const data = await kalshiGet(
      `/markets?series_ticker=KXEPLGAME&status=open&limit=5`,
      kalshiKey
    );
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// ═════════════════════════════════════════════════════════════════════════════
// START
// ═════════════════════════════════════════════════════════════════════════════

app.listen(PORT, () => {
  console.log(`\n✅  EPL Predictions server running at http://localhost:${PORT}\n`);
  console.log('   Routes:');
  console.log('     GET /api/kalshi-epl              → Kalshi EPL match markets');
  console.log('     GET /api/h2h?home=X&away=Y       → API-Football H2H stats');
  console.log('     GET /api/kalshi-debug            → Raw Kalshi market dump\n');
});
