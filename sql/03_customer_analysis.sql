-- =========================================
-- CUSTOMER ANALYSIS
-- =========================================


-- 1. Customer performance

SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    c.country,
    c.city,
    COUNT(DISTINCT f.order_id) AS orders,
    SUM(f.quantity) AS units_bought,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit,
    ROUND(
        SUM(f.revenue) / NULLIF(COUNT(DISTINCT f.order_id), 0),
        2
    ) AS average_order_value
FROM warehouse.fact_sales f
JOIN warehouse.dim_customer c
    ON c.customer_key = f.customer_key
GROUP BY
    c.customer_id,
    c.first_name,
    c.last_name,
    c.country,
    c.city
ORDER BY revenue DESC;


-- 2. Top 20 customers by revenue

SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    c.country,
    COUNT(DISTINCT f.order_id) AS orders,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit
FROM warehouse.fact_sales f
JOIN warehouse.dim_customer c
    ON c.customer_key = f.customer_key
GROUP BY
    c.customer_id,
    c.first_name,
    c.last_name,
    c.country
ORDER BY revenue DESC
LIMIT 20;


-- 3. Customers by country

SELECT
    c.country,
    COUNT(DISTINCT c.customer_key) AS customers,
    COUNT(DISTINCT f.order_id) AS orders,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit
FROM warehouse.dim_customer c
LEFT JOIN warehouse.fact_sales f
    ON f.customer_key = c.customer_key
GROUP BY c.country
ORDER BY revenue DESC;


-- 4. Customer lifetime value proxy

SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    MIN(d.full_date) AS first_order_date,
    MAX(d.full_date) AS last_order_date,
    COUNT(DISTINCT f.order_id) AS order_count,
    ROUND(SUM(f.revenue), 2) AS lifetime_revenue,
    ROUND(SUM(f.profit), 2) AS lifetime_profit
FROM warehouse.fact_sales f
JOIN warehouse.dim_customer c
    ON c.customer_key = f.customer_key
JOIN warehouse.dim_date d
    ON d.date_key = f.date_key
GROUP BY
    c.customer_id,
    c.first_name,
    c.last_name
ORDER BY lifetime_revenue DESC;


-- 5. Customer order frequency

SELECT
    order_count,
    COUNT(*) AS customers
FROM (
    SELECT
        customer_key,
        COUNT(DISTINCT order_id) AS order_count
    FROM warehouse.fact_sales
    GROUP BY customer_key
) customer_orders
GROUP BY order_count
ORDER BY order_count;


-- 6. RFM base dataset

SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    MAX(d.full_date) AS last_order_date,
    COUNT(DISTINCT f.order_id) AS frequency,
    ROUND(SUM(f.revenue), 2) AS monetary
FROM warehouse.fact_sales f
JOIN warehouse.dim_customer c
    ON c.customer_key = f.customer_key
JOIN warehouse.dim_date d
    ON d.date_key = f.date_key
GROUP BY
    c.customer_id,
    c.first_name,
    c.last_name
ORDER BY monetary DESC;