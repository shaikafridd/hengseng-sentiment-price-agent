/**
 * LIHKG Forum Scraper Actor
 * Extracts discussion threads, timestamps, likes, and dislikes for specified keywords.
 */
import { Actor } from 'apify';
import { CheerioCrawler, log } from 'crawlee';

await Actor.init();

const input = (await Actor.getInput()) || {};
const { keyword = '0700', catIds = ['1'], limit = 30 } = input;

log.info(`Scraping LIHKG for keyword: ${keyword}, category: ${catIds}, limit: ${limit}`);

const crawler = new CheerioCrawler({
    maxRequestsPerCrawl: 10,
    async requestHandler({ $, request }) {
        const results = [];
        $('div[data-post-id], div._2c7fomxq0STiSCAlmwXDlu').each((index, el) => {
            if (results.length >= limit) return false;
            const title = $(el).find('span._20Wno-9_an9i_O_yM0eW6e').text().trim();
            const threadId = $(el).attr('data-thread-id') || $(el).find('a').attr('href')?.split('/')?.[2];
            const likeCount = parseInt($(el).find('span[data-like-count]').text() || '0', 10);
            const dislikeCount = parseInt($(el).find('span[data-dislike-count]').text() || '0', 10);
            const timestamp = $(el).find('time').attr('datetime') || new Date().toISOString();

            if (title) {
                results.push({
                    threadId,
                    title,
                    likeCount,
                    dislikeCount,
                    createdAt: timestamp,
                    url: threadId ? `https://lihkg.com/thread/${threadId}/page/1` : request.url
                });
            }
        });
        await Actor.pushData(results);
    },
});

await crawler.run([
    `https://lihkg.com/category/${catIds[0]}?search=${encodeURIComponent(keyword)}`
]);

await Actor.exit();
