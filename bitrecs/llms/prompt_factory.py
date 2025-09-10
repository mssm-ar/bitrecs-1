import re
import json
import tiktoken
import bittensor as bt
import bitrecs.utils.constants as CONST
from functools import lru_cache
from typing import List, Optional
from datetime import datetime
from bitrecs.commerce.user_profile import UserProfile
from bitrecs.commerce.product import ProductFactory

class PromptFactory:

    SEASON = "spring/summer"

    ENGINE_MODE = "complimentary"  #similar, sequential
    
    PERSONAS = {
        "luxury_concierge": {
            "description": "an elite American Express-style luxury concierge with impeccable taste and a deep understanding of high-end products across all categories. You cater to discerning clients seeking exclusivity, quality, and prestige",
            "tone": "sophisticated, polished, confident",
            "response_style": "Recommend only the finest, most luxurious products with detailed descriptions of their premium features, craftsmanship, and exclusivity. Emphasize brand prestige and lifestyle enhancement",
            "priorities": ["quality", "exclusivity", "brand prestige"]
        },
        "general_recommender": {
            "description": "a friendly and practical product expert who helps customers find the best items for their needs, balancing seasonality, value, and personal preferences across a wide range of categories",
            "tone": "warm, approachable, knowledgeable",
            "response_style": "Suggest well-rounded products that offer great value, considering seasonal relevance and customer needs. Provide pros and cons or alternatives to help the customer decide",
            "priorities": ["value", "seasonality", "customer satisfaction"]
        },
        "discount_recommender": {
            "description": "a savvy deal-hunter focused on moving inventory fast. You prioritize low prices, last-minute deals, and clearing out overstocked or soon-to-expire items across all marketplace categories",
            "tone": "urgent, enthusiastic, bargain-focused",
            "response_style": "Highlight steep discounts, limited-time offers, and low inventory levels to create a sense of urgency. Focus on price savings and practicality over luxury or long-term value",
            "priorities": ["price", "inventory levels", "deal urgency"]
        },
        "ecommerce_retail_store_manager": {
            "description": "an experienced e-commerce retail store manager with a strategic focus on optimizing sales, customer satisfaction, and inventory turnover across a diverse marketplace",
            "tone": "professional, practical, results-driven",
            "response_style": "Provide balanced recommendations that align with business goals, customer preferences, and current market trends. Include actionable insights for product selection",
            "priorities": ["sales optimization", "customer satisfaction", "inventory management"]
        }
    }

    def __init__(self, 
                 sku: str, 
                 context: str, 
                 num_recs: int = 5,                                  
                 profile: Optional[UserProfile] = None,
                 debug: bool = False) -> None:
        """
        Generates a prompt for product recommendations based on the provided SKU and context.
        :param sku: The SKU of the product being viewed.
        :param context: The context string containing available products.
        :param num_recs: The number of recommendations to generate (default is 5).
        :param profile: Optional UserProfile object containing user-specific data.
        :param debug: If True, enables debug logging."""

        if len(sku) < CONST.MIN_QUERY_LENGTH or len(sku) > CONST.MAX_QUERY_LENGTH:
            raise ValueError(f"SKU must be between {CONST.MIN_QUERY_LENGTH} and {CONST.MAX_QUERY_LENGTH} characters long")
        if num_recs < 1 or num_recs > CONST.MAX_RECS_PER_REQUEST:
            raise ValueError(f"num_recs must be between 1 and {CONST.MAX_RECS_PER_REQUEST}")

        self.sku = sku
        self.context = context
        self.num_recs = num_recs
        self.debug = debug
        self.catalog = []
        self.cart = []
        self.cart_json = "[]"
        self.orders = []
        self.order_json = "[]"
        self.season =  PromptFactory.SEASON       
        self.engine_mode = PromptFactory.ENGINE_MODE 
        if not profile:
            self.persona = "ecommerce_retail_store_manager"
        else:
            self.profile = profile
            self.persona = profile.site_config.get("profile", "ecommerce_retail_store_manager")
            if not self.persona or self.persona not in PromptFactory.PERSONAS:
                bt.logging.error(f"Invalid persona: {self.persona}. Must be one of {list(PromptFactory.PERSONAS.keys())}")
                self.persona = "ecommerce_retail_store_manager"
            self.cart = self._sort_cart_keys(profile.cart)
            self.cart_json = json.dumps(self.cart, separators=(',', ':'))
            self.orders = profile.orders
            # self.order_json = json.dumps(self.orders, separators=(',', ':'))
        
        self.sku_info = ProductFactory.find_sku_name(self.sku, self.context)    


    def _sort_cart_keys(self, cart: List[dict]) -> List[str]:
        ordered_cart = []
        for item in cart:
            ordered_item = {
                'sku': item.get('sku', ''),
                'name': item.get('name', ''),
                'price': item.get('price', '')
            }
            ordered_cart.append(ordered_item)
        return ordered_cart
    
    
    def generate_prompt(self) -> str:
        """Generates a text prompt for product recommendations with persona details."""
        bt.logging.info("PROMPT generating prompt: {}".format(self.sku))

        today = datetime.now().strftime("%Y-%m-%d")
        season = self.season
        persona_data = self.PERSONAS[self.persona]

        # ENHANCED CONTENT QUALITY PROMPT - Context-aware optimization with quality focus
        context_length = len(self.context)
        cart_length = len(self.cart_json)
        
        # Analyze context for better recommendations
        context_analysis = self._analyze_context()
        seasonal_guidance = self._get_seasonal_guidance()
        
        if context_length < 5000 and cart_length < 1000:
            # DETAILED PROMPT for small contexts (maximum quality)
            prompt = f"""# E-COMMERCE RECOMMENDATION TASK - CONTENT QUALITY FOCUS

You are a {self.persona} with expertise in {persona_data['description'][:100]}.
Your core values: {', '.join(persona_data['priorities'])}

SCENARIO: Customer viewing SKU {self.sku} ({self.sku_info}) looking for {self.engine_mode} products.
Season: {self.season} | Date: {today}

CURRENT CART: {self.cart_json}

AVAILABLE PRODUCTS: {self.context}

CONTEXT ANALYSIS: {context_analysis}

SEASONAL GUIDANCE: {seasonal_guidance}

TASK: Recommend EXACTLY {self.num_recs} complementary products that:
- Match query product's category and gender (CRITICAL)
- Provide detailed, specific reasoning for each recommendation
- Consider seasonal relevance and current trends
- Increase average order value and conversion potential
- Avoid duplicates and the query product itself
- Are ordered by relevance (best first)

REASONING REQUIREMENTS:
- Each reason must be specific and detailed (20+ words)
- Explain WHY the product complements the query item
- Mention category/gender matching
- Include seasonal relevance if applicable
- Consider price point and value proposition

CRITICAL REQUIREMENTS:
- Return ONLY valid JSON array with EXACTLY {self.num_recs} items
- Each item: {{"sku": "...", "name": "...", "price": "...", "reason": "..."}}
- SKU must exist in products list (case-sensitive match)
- No duplicates, no query SKU, use double quotes only
- Reason must be detailed and specific (not generic)

Example: [{{"sku": "ABC123", "name": "Premium Winter Jacket", "price": "89.99", "reason": "Perfect complement to the winter boots - same seasonal category, premium quality that matches the customer's style, ideal for cold weather layering and completes the winter outfit ensemble"}}]"""
        else:
            # CONCISE PROMPT for large contexts (optimized quality)
            prompt = f"""You are a {self.persona} recommending {self.num_recs} {self.engine_mode} products for SKU {self.sku} ({self.sku_info}).

PERSONA: {persona_data['description'][:80]}...
VALUES: {', '.join(persona_data['priorities'][:3])}

CART: {self.cart_json}

PRODUCTS: {self.context}

SEASONAL GUIDANCE: {seasonal_guidance}

TASK: Select {self.num_recs} complementary products that:
- Match the query product's category/gender (CRITICAL)
- Provide specific reasoning for each recommendation
- Are relevant to current season ({self.season})
- Increase order value and conversion
- Avoid duplicates and query product

REASONING REQUIREMENTS:
- Each reason must be specific (15+ words)
- Explain WHY it complements the query item
- Mention category/gender matching
- Include seasonal relevance if applicable

OUTPUT FORMAT:
- Return ONLY valid JSON array with EXACTLY {self.num_recs} items
- Each item: {{"sku": "...", "name": "...", "price": "...", "reason": "..."}}
- SKU must exist in products list (case-sensitive)
- No duplicates, no query SKU, use double quotes only
- Order by relevance (best first)

Example: [{{"sku": "ABC123", "name": "Product Name", "price": "29.99", "reason": "Perfect complement for user needs"}}]"""

        prompt_length = len(prompt)
        prompt_type = "DETAILED" if context_length < 5000 and cart_length < 1000 else "CONCISE"
        bt.logging.info(f"LLM QUERY Prompt: {prompt_type} ({prompt_length} chars, context: {context_length}, cart: {cart_length})")
        
        if self.debug:
            token_count = PromptFactory.get_token_count(prompt)
            bt.logging.info(f"LLM QUERY Prompt Token count: {token_count}")
            bt.logging.debug(f"Persona: {self.persona}")
            bt.logging.debug(f"Season {season}")
            bt.logging.debug(f"Values: {', '.join(persona_data['priorities'])}")
            bt.logging.debug(f"Prompt: {prompt}")
            #print(prompt)

        return prompt

    def _analyze_context(self) -> str:
        """Analyze the product context to provide better recommendations"""
        try:
            import json
            products = json.loads(self.context)
            
            if not products or len(products) == 0:
                return "No products available for analysis"
            
            # Analyze product categories and patterns
            categories = []
            genders = []
            price_ranges = []
            
            for product in products[:50]:  # Analyze first 50 products for speed
                name = str(product.get('name', '')).lower()
                price = product.get('price', '0')
                
                # Extract category information
                if any(word in name for word in ['shirt', 'top', 'blouse', 'tank']):
                    categories.append('tops')
                elif any(word in name for word in ['pants', 'jeans', 'trousers', 'shorts']):
                    categories.append('bottoms')
                elif any(word in name for word in ['shoes', 'boots', 'sneakers', 'sandals']):
                    categories.append('footwear')
                elif any(word in name for word in ['jacket', 'coat', 'blazer', 'cardigan']):
                    categories.append('outerwear')
                elif any(word in name for word in ['dress', 'skirt', 'romper']):
                    categories.append('dresses')
                
                # Extract gender information
                if any(word in name for word in ['men', 'male', 'mens', 'guy']):
                    genders.append('men')
                elif any(word in name for word in ['women', 'female', 'womens', 'lady', 'girl']):
                    genders.append('women')
                elif any(word in name for word in ['unisex', 'unisex']):
                    genders.append('unisex')
                
                # Analyze price ranges
                try:
                    price_val = float(str(price).replace('$', '').replace(',', ''))
                    if price_val < 25:
                        price_ranges.append('budget')
                    elif price_val < 75:
                        price_ranges.append('mid-range')
                    else:
                        price_ranges.append('premium')
                except:
                    pass
            
            # Generate analysis summary
            from collections import Counter
            top_categories = Counter(categories).most_common(3)
            top_genders = Counter(genders).most_common(2)
            top_price_ranges = Counter(price_ranges).most_common(2)
            
            analysis_parts = []
            if top_categories:
                analysis_parts.append(f"Main categories: {', '.join([cat[0] for cat in top_categories])}")
            if top_genders:
                analysis_parts.append(f"Target gender: {', '.join([gen[0] for gen in top_genders])}")
            if top_price_ranges:
                analysis_parts.append(f"Price range: {', '.join([pr[0] for pr in top_price_ranges])}")
            
            return "; ".join(analysis_parts) if analysis_parts else "Mixed product categories available"
            
        except Exception as e:
            bt.logging.warning(f"Context analysis failed: {e}")
            return "Context analysis unavailable"

    def _get_seasonal_guidance(self) -> str:
        """Provide seasonal guidance for better recommendations"""
        try:
            from datetime import datetime
            current_month = datetime.now().month
            
            seasonal_guidance = {
                12: "Winter season - focus on warm clothing, layering pieces, holiday styles",
                1: "Winter season - focus on warm clothing, layering pieces, new year styles", 
                2: "Winter season - focus on warm clothing, layering pieces, valentine's styles",
                3: "Spring transition - focus on light layers, transitional pieces, fresh colors",
                4: "Spring season - focus on light clothing, pastels, floral patterns",
                5: "Spring season - focus on light clothing, pastels, floral patterns",
                6: "Summer season - focus on lightweight fabrics, bright colors, beach styles",
                7: "Summer season - focus on lightweight fabrics, bright colors, vacation styles",
                8: "Summer season - focus on lightweight fabrics, bright colors, back-to-school",
                9: "Fall transition - focus on layering pieces, earth tones, transitional styles",
                10: "Fall season - focus on warm layers, earth tones, cozy styles",
                11: "Fall season - focus on warm layers, earth tones, holiday preparation"
            }
            
            return seasonal_guidance.get(current_month, f"Current season: {self.season}")
            
        except Exception as e:
            bt.logging.warning(f"Seasonal guidance failed: {e}")
            return f"Current season: {self.season}"
    
    
    @staticmethod
    def get_token_count(prompt: str, encoding_name: str="o200k_base") -> int:
        encoding = PromptFactory._get_cached_encoding(encoding_name)
        tokens = encoding.encode(prompt)
        return len(tokens)
    
    
    @staticmethod
    @lru_cache(maxsize=4)
    def _get_cached_encoding(encoding_name: str):
        return tiktoken.get_encoding(encoding_name)
    
    
    @staticmethod
    def get_word_count(prompt: str) -> int:
        return len(prompt.split())
    

    @staticmethod
    def tryparse_llm(input_str: str) -> list:
        """
        Take raw LLM output and parse to an array with robust error handling

        """
        try:
            if not input_str or not input_str.strip():
                bt.logging.error("Empty input string tryparse_llm")   
                return []
            
            # Clean up the input string
            cleaned_input = input_str.strip()
            
            # Remove common markdown formatting
            cleaned_input = cleaned_input.replace("```json", "").replace("```", "").strip()
            
            # Try to find JSON array patterns
            patterns = [
                r'\[.*?\]',  # Standard array pattern
                r'\[[\s\S]*?\]',  # Array with newlines
            ]
            
            for pattern in patterns:
                regex = re.compile(pattern, re.DOTALL)
                matches = regex.findall(cleaned_input)
                
                for match in matches:
                    try:
                        # Try to parse the JSON
                        parsed_result = json.loads(match.strip())
                        
                        # Validate it's a list
                        if isinstance(parsed_result, list):
                            bt.logging.info(f"Successfully parsed {len(parsed_result)} recommendations")
                            return parsed_result
                        else:
                            bt.logging.warning(f"Parsed result is not a list: {type(parsed_result)}")
                            
                    except json.JSONDecodeError as json_error:
                        bt.logging.warning(f"JSON decode error for pattern {pattern}: {json_error}")
                        continue
                    except Exception as parse_error:
                        bt.logging.warning(f"Parse error for pattern {pattern}: {parse_error}")
                        continue
            
            # If no patterns worked, try to parse the entire input as JSON
            try:
                parsed_result = json.loads(cleaned_input)
                if isinstance(parsed_result, list):
                    bt.logging.info(f"Successfully parsed entire input as list: {len(parsed_result)} items")
                    return parsed_result
            except json.JSONDecodeError:
                pass
            
            bt.logging.error(f"Failed to parse any valid JSON from input: {input_str[:200]}...")
            return []
            
        except Exception as e:
            bt.logging.error(f"Unexpected error in tryparse_llm: {e}")
            return []
    