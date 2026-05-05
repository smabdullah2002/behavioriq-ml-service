# ML Service Test Cases

Simple endpoint test cases for the ML service.

## Intent Endpoint

`POST /ml/intent-score`

| Case | Request Body | Expected |
|---|---|---|
| Empty session | `{ "product_visit_count": 0, "time_on_product_page": 0, "cart_add_events": 0, "search_to_view_ratio": 0, "price_range_affinity": 0, "session_recency": 0 }` | Score should be `0` or very close to `0`, with all contributions near `0`. |
| Light browsing | `{ "product_visit_count": 2, "time_on_product_page": 18, "cart_add_events": 0, "search_to_view_ratio": 0.2, "price_range_affinity": 0.3, "session_recency": 24 }` | Low score. Product views and page time should contribute a little. |
| Strong purchase intent | `{ "product_visit_count": 12, "time_on_product_page": 95, "cart_add_events": 3, "search_to_view_ratio": 0.8, "price_range_affinity": 0.7, "session_recency": 2 }` | High score. Cart adds, page time, and recent activity should drive the score up. |
| Cart-heavy user | `{ "product_visit_count": 5, "time_on_product_page": 20, "cart_add_events": 5, "search_to_view_ratio": 0.4, "price_range_affinity": 0.5, "session_recency": 1 }` | Very high score. `cart_add_events` should be one of the biggest contributions. |
| Out-of-range input | `{ "product_visit_count": 50, "time_on_product_page": 1000, "cart_add_events": 20, "search_to_view_ratio": 2.5, "price_range_affinity": 120, "session_recency": -5 }` | Service should clamp/normalize values and still return a score in the `0-100` range. |

## Churn Endpoint

`POST /ml/churn-predict`

| Case | Request Body | Expected |
|---|---|---|
| Very recent buyer | `{ "days_since_last_purchase": 2, "total_order_count": 12, "avg_order_value": 120 }` | Low churn probability. Frequent and recent buyers should look healthy. |
| Moderate customer | `{ "days_since_last_purchase": 20, "total_order_count": 5, "avg_order_value": 75 }` | Medium churn probability. Balanced profile. |
| At-risk customer | `{ "days_since_last_purchase": 60, "total_order_count": 2, "avg_order_value": 40 }` | Higher churn probability because recency and frequency are weak. |
| Very high risk | `{ "days_since_last_purchase": 180, "total_order_count": 1, "avg_order_value": 20 }` | Churn probability should be high. |
| Strong loyal customer | `{ "days_since_last_purchase": 1, "total_order_count": 30, "avg_order_value": 150 }` | Very low churn probability. |

## Quick Checks

- Intent score should always be between `0` and `100`.
- Intent response should include `contributions` for all six input features.
- Churn probability should always be between `0` and `1`.
- Lower `days_since_last_purchase` should generally reduce churn.
- Higher `total_order_count` should generally reduce churn.