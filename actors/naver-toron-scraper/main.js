/**
 * Naver Finance Toronsil (종목토론실) Scraper Actor
 * Extracts discussion board titles, dates, view counts, and upvotes/downvotes.
 */
import { Actor } from 'apify';
import { CheerioCrawler, log } from 'crawlee';

await Actor.init();

const input = (await Actor.getInput()) || {};
const { code = '005930', maxPages = 3 } = input;

log.info(`Scraping Naver Finance Toronsil for ticker code: ${code}, maxPages: ${maxPages}`);

const crawler = new CheerioCrawler({
    maxRequestsPerCrawl: maxPages,
    async requestHandler({ $, request }) {
        const items = [];
        $('table.type2 tbody tr').each((i, el) => {
            const titleEl = $(el).find('td.title a');
            if (titleEl.length > 0) {
                const title = titleEl.text().trim();
                const href = titleEl.attr('href');
                const date = $(el).find('td:nth-child(2) span').text().trim();
                const views = $(el).find('td:nth-child(4) span').text().trim();
                const upvotes = $(el).find('td:nth-child(5) strong').text().trim();
                const downvotes = $(el).find('td:nth-child(6) strong').text().trim();
                items.push({
                    title,
                    href: `https://finance.naver.com${href}`,
                    date,
                    views: parseInt(views.replace(/,/g, '') || '0', 10),
                    upvotes: parseInt(upvotes || '0', 10),
                    downvotes: parseInt(downvotes || '0', 10),
                });
            }
        });
        await Actor.pushData(items);
    },
});

const urls = [];
for (let p = 1; p <= maxPages; p++) {
    urls.push(`https://finance.naver.com/item/board.naver?code=${code}&page=${p}`);
}

await crawler.run(urls);

await Actor.exit();
