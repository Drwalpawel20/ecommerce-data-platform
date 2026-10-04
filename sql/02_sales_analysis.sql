-- =========================================
-- SALES ANALYSIS
-- =========================================


-- 1. Overall KPIs

SELECT
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(quantity) AS units_sold,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND(SUM(cost), 2) AS total_cost,
    ROUND(SUM(profit), 2) AS total_profit,
    ROUND(
        SUM(profit) / NULLIF(SUM(revenue), 0) * 100,
        2
    ) AS profit_margin_percent,
    ROUND(
        SUM(revenue) / NULLIF(COUNT(DISTINCT order_id), 0),
        2
    ) AS average_order_value
FROM warehouse.fact_sales;


-- 2. Revenue by month

SELECT
    d.year_number,
    d.month_number,
    d.month_name,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit,
    COUNT(DISTINCT f.order_id) AS orders
FROM warehouse.fact_sales f
JOIN warehouse.dim_date d
    ON d.date_key = f.date_key
GROUP BY
    d.year_number,
    d.month_number,
    d.month_name
ORDER BY
    d.year_number,
    d.month_number;


-- 3. Revenue by year

SELECT
    d.year_number,
    COUNT(DISTINCT f.order_id) AS orders,
    SUM(f.quantity) AS units_sold,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit,
    ROUND(
        SUM(f.profit) / NULLIF(SUM(f.revenue), 0) * 100,
        2
    ) AS profit_margin_percent
FROM warehouse.fact_sales f
JOIN warehouse.dim_date d
    ON d.date_key = f.date_key
GROUP BY d.year_number
ORDER BY d.year_number;


-- 4. Revenue by order status

SELECT
    order_status,
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(revenue), 2) AS revenue,
    ROUND(SUM(profit), 2) AS profit
FROM warehouse.fact_sales
GROUP BY order_status
ORDER BY revenue DESC;


-- 5. Revenue by country

SELECT
    c.country,
    COUNT(DISTINCT f.order_id) AS orders,
    COUNT(DISTINCT c.customer_key) AS customers,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit
FROM warehouse.fact_sales f
JOIN warehouse.dim_customer c
    ON c.customer_key = f.customer_key
GROUP BY c.country
ORDER BY revenue DESC;


-- 6. Revenue by employee

SELECT
    e.employee_id,
    e.first_name,
    e.last_name,
    e.department,
    COUNT(DISTINCT f.order_id) AS orders,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit
FROM warehouse.fact_sales f
JOIN warehouse.dim_employee e
    ON e.employee_key = f.employee_key
GROUP BY
    e.employee_id,
    e.first_name,
    e.last_name,
    e.department
ORDER BY revenue DESC;


-- 7. Daily sales

SELECT
    d.full_date,
    COUNT(DISTINCT f.order_id) AS orders,
    SUM(f.quantity) AS units_sold,
    ROUND(SUM(f.revenue), 2) AS revenue,
    ROUND(SUM(f.profit), 2) AS profit
FROM warehouse.fact_sales f
JOIN warehouse.dim_date d
    ON d.date_key = f.date_key
GROUP BY d.full_date
ORDER BY d.full_date;