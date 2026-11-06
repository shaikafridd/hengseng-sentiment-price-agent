/**
 * AAStocks Scraper Actor
 * Extracts retail posts, timestamps, and usernames from AAStocks stock forums.
 */
import { Actor } from 'apify';
import { CheerioCrawler, log } from 'crawlee';

await Actor.init();

const input = (await Actor.getInput()) || {};
const { symbol = '0700', limit = 30 } = input;

log.info(`Scraping AAStocks Forum for symbol: ${symbol}, limit: ${limit}`);

const crawler = new CheerioCrawler({
    maxRequestsPerCrawl: 5,
    async requestHandler({ $, request }) {
        const posts = [];
        $('.forum-content, .contentBox, tr.forumRow').each((i, el) => {
            if (posts.length >= limit) return false;
            const title = $(el).find('.newsTitle, .title').text().trim();
            const body = $(el).find('.desc, .content, .comment').text().trim();
            const dateStr = $(el).find('.date, .datetime').text().trim();
            const author = $(el).find('.author, .username').text().trim() || 'Anonymous';

            if (title || body) {
                posts.push({
                    title,
                    body,
                    dateStr,
                    author,
                    url: request.url
                });
            }
        });
        await Actor.pushData(posts);
    },
});

await crawler.run([
    `http://www.aastocks.com/tc/stocks/analysis/stock-forum.aspx?symbol=${symbol}`
]);

await Actor.exit();
