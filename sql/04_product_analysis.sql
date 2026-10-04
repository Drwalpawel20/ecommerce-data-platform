-- =========================================
-- PRODUCT ANALYSIS
-- =========================================


-- 1. Product performance

SELECT
    p.product_id,
    p.product_name,
    c.category_name,
    s.supplier_name,
    COUNT(DISTINCT f.order_id) AS orders,
    SUM(f.quantity) AS units_sold,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.cost), 2) AS cost,
    ROUND(SUM(f.profit), 2) AS profit,
    ROUND(
        SUM(f.profit) / NULLIF(SUM(f.revenue), 0) * 100,
        2
    ) AS margin_percent
FROM warehouse.fact_sales f
JOIN warehouse.dim_product p
    ON p.product_key = f.product_key
JOIN warehouse.dim_category c
    ON c.category_key = p.category_key
JOIN warehouse.dim_supplier s
    ON s.supplier_key = p.supplier_key
GROUP BY
    p.product_id,
    p.product_name,
    c.category_name,
    s.supplier_name
ORDER BY revenue DESC;


-- 2. Top 20 products by revenue

SELECT
    p.product_id,
    p.product_name,
    c.category_name,
    SUM(f.quantity) AS units_sold,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit
FROM warehouse.fact_sales f
JOIN warehouse.dim_product p
    ON p.product_key = f.product_key
JOIN warehouse.dim_category c
    ON c.category_key = p.category_key
GROUP BY
    p.product_id,
    p.product_name,
    c.category_name
ORDER BY revenue DESC
LIMIT 20;


-- 3. Top 20 products by profit

SELECT
    p.product_id,
    p.product_name,
    c.category_name,
    SUM(f.quantity) AS units_sold,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit
FROM warehouse.fact_sales f
JOIN warehouse.dim_product p
    ON p.product_key = f.product_key
JOIN warehouse.dim_category c
    ON c.category_key = p.category_key
GROUP BY
    p.product_id,
    p.product_name,
    c.category_name
ORDER BY profit DESC
LIMIT 20;


-- 4. Category performance

SELECT
    c.category_id,
    c.category_name,
    COUNT(DISTINCT p.product_id) AS products,
    SUM(f.quantity) AS units_sold,
    COUNT(DISTINCT f.order_id) AS orders,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit,
    ROUND(
        SUM(f.profit) / NULLIF(SUM(f.revenue), 0) * 100,
        2
    ) AS margin_percent
FROM warehouse.fact_sales f
JOIN warehouse.dim_product p
    ON p.product_key = f.product_key
JOIN warehouse.dim_category c
    ON c.category_key = p.category_key
GROUP BY
    c.category_id,
    c.category_name
ORDER BY revenue DESC;


-- 5. Supplier performance

SELECT
    s.supplier_id,
    s.supplier_name,
    s.country,
    COUNT(DISTINCT p.product_id) AS products,
    SUM(f.quantity) AS units_sold,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit
FROM warehouse.fact_sales f
JOIN warehouse.dim_product p
    ON p.product_key = f.product_key
JOIN warehouse.dim_supplier s
    ON s.supplier_key = p.supplier_key
GROUP BY
    s.supplier_id,
    s.supplier_name,
    s.country
ORDER BY revenue DESC;


-- 6. Category + year analysis

SELECT
    d.year_number,
    c.category_name,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit,
    SUM(f.quantity) AS units_sold
FROM warehouse.fact_sales f
JOIN warehouse.dim_product p
    ON p.product_key = f.product_key
JOIN warehouse.dim_category c
    ON c.category_key = p.category_key
JOIN warehouse.dim_date d
    ON d.date_key = f.date_key
GROUP BY
    d.year_number,
    c.category_name
ORDER BY
    d.year_number,
    revenue DESC;