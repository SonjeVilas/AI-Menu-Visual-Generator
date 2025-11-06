import streamlit as st
import base64
from datetime import datetime
import json
import re
import requests
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="GenAILearnivserse Project 18: AI Menu Visual Generator",
    layout="wide",
    page_icon="👀"
)

# API Key Input Section
st.markdown("### 🔑 Gemini API Configuration")
api_key_input = st.text_input(
    "Enter your Gemini API Key",
    type="password",
    placeholder="AIza...",
    help="Get your API key from https://aistudio.google.com/app/apikey"
)

# Initialize Gemini configuration with UI input
client = None
model_name = "gemini-2.5-flash"
if api_key_input:
    try:
        genai.configure(api_key=api_key_input)
        client = genai.GenerativeModel(model_name)
        st.success("✅ API key provided successfully!")
    except Exception as e:
        st.error(f"❌ Invalid API key: {str(e)}")
        st.stop()
        

def extract_menu_text_from_image(image_file):
    """Extract text from an uploaded menu image using Gemini with a strict JSON response."""
    try:
        image_bytes = image_file.read()
        image_file.seek(0)

        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )

        schema_instruction = (
            "You are an OCR+structure extractor. Return ONLY JSON with this structure: "
            "{\n  \"restaurant_name\": string,\n  \"cuisine_type\": string,\n  \"categories\": {\n    <CategoryName>: [\n      {\n        \"name\": string,\n        \"description\": string (optional),\n        \"price\": string (optional)\n      }\n    ]\n  }\n}\n"
            "- Never invent placeholders like 'Item 1', 'Item 2'. If unsure, leave name empty and keep text in description.\n"
            "- Normalize prices exactly as seen (e.g., '70/-').\n"
            "- If no categories, use a single category named 'Uncategorized'."
        )

        mime_type = getattr(image_file, "type", None) or "image/jpeg"
        parts = [
            schema_instruction,
            {"mime_type": mime_type, "data": image_bytes},
        ]
        response = client.generate_content(parts, generation_config=generation_config)
        text = (response.text or "").strip()
        # Direct JSON expected
        try:
            data = json.loads(text)
            # Fallback: if many placeholder names like "Item 1" are present, try a refinement step
            data = _maybe_refine_item_names_with_gemini(data, image_bytes, mime_type)
            return data
        except Exception:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                data = json.loads(text[json_start:json_end])
                data = _maybe_refine_item_names_with_gemini(data, image_bytes, mime_type)
                return data
            return {"error": "Could not parse menu structure", "raw_text": text}
    except Exception as e:
        return {"error": f"Failed to extract menu text: {str(e)}"}
        

def get_nutritional_info(dish_name, cuisine_type):
    """Get estimated nutritional information for a dish using Gemini."""
    try:
        prompt = (
            f"For the {cuisine_type} dish \"{dish_name}\", provide estimated nutritional information in JSON format only with keys: "
            f"calories, protein, carbs, fat, fiber. Base your estimate on typical restaurant portions and ingredients."
        )
        response = client.generate_content(prompt)
        nutrition_text = (response.text or "").strip()
        json_start = nutrition_text.find('{')
        json_end = nutrition_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = nutrition_text[json_start:json_end]
            return json.loads(json_str)
        return {"calories": "N/A", "protein": "N/A", "carbs": "N/A", "fat": "N/A", "fiber": "N/A"}
    except Exception:
        return {"calories": "N/A", "protein": "N/A", "carbs": "N/A", "fat": "N/A", "fiber": "N/A"}
        

def normalize_menu_data(raw_data):
    """Normalize LLM output so that menu_data['categories'] is a dict of category -> list of items."""
    try:
        data = raw_data if isinstance(raw_data, dict) else {"categories": raw_data}

        restaurant_name = data.get("restaurant_name") or "Unknown Restaurant"
        cuisine_type = data.get("cuisine_type") or "Unknown"

        categories = data.get("categories")
        grouped = {}

        if isinstance(categories, dict):
            for cat, items in categories.items():
                if isinstance(items, list):
                    # Ensure each item has a name; try to infer from text/description when missing
                    fixed = []
                    for it in items:
                        if isinstance(it, dict) and not it.get("name"):
                            inferred = _infer_name_from_text(it)
                            if inferred:
                                it["name"] = inferred
                        fixed.append(it)
                    grouped[cat] = fixed
                else:
                    grouped[cat] = [items]
        elif isinstance(categories, list):
            for entry in categories:
                if isinstance(entry, dict):
                    if "items" in entry and isinstance(entry.get("items"), list):
                        cat_name = entry.get("name") or entry.get("category") or "Uncategorized"
                        items = entry.get("items") or []
                        fixed = []
                        for it in items:
                            if isinstance(it, dict) and not it.get("name"):
                                inferred = _infer_name_from_text(it)
                                if inferred:
                                    it["name"] = inferred
                            fixed.append(it)
                        grouped.setdefault(cat_name, []).extend(fixed)
                    else:
                        cat_name = entry.get("category") or "Uncategorized"
                        if not entry.get("name"):
                            inferred = _infer_name_from_text(entry)
                            if inferred:
                                entry["name"] = inferred
                        grouped.setdefault(cat_name, []).append(entry)
                else:
                    grouped.setdefault("Menu", []).append({"name": str(entry)})
        else:
            items = data.get("items") or data.get("menu_items") or []
            if isinstance(items, list):
                for entry in items:
                    if isinstance(entry, dict):
                        cat = entry.get("category") or "Uncategorized"
                        if not entry.get("name"):
                            inferred = _infer_name_from_text(entry)
                            if inferred:
                                entry["name"] = inferred
                        grouped.setdefault(cat, []).append(entry)
                    else:
                        grouped.setdefault("Menu", []).append({"name": str(entry)})
            else:
                grouped["Menu"] = []

        if not grouped:
            grouped["Menu"] = []

        return {
            "restaurant_name": restaurant_name,
            "cuisine_type": cuisine_type,
            "categories": grouped,
        }
    except Exception:
        return {
            "restaurant_name": "Unknown Restaurant",
            "cuisine_type": "Unknown",
            "categories": {"Menu": []},
        }

def _item_name(item, idx=0):
    if isinstance(item, dict):
        return str(item.get("name") or item.get("title") or item.get("item") or item.get("dish") or _infer_name_from_text(item) or f"Item {idx+1}")
    return str(item)

def _item_field(item, key, default=""):
    return item.get(key, default) if isinstance(item, dict) else default

def _infer_name_from_text(item: dict):
    """Try to infer a dish name from description/text fields when 'name' is missing."""
    text_sources = []
    for k in ("text", "description", "details", "raw"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            text_sources.append(v.strip())
    if not text_sources:
        return None
    text = text_sources[0]
    # Heuristics: split on price patterns and punctuation, take leading phrase
    import re as _re
    text = _re.split(r"\s*(?:\d+\s*/?-?\s*\/?-?\s*|\d+\.?\d*\s*(?:INR|Rs\.?|₹|\$)|-\s*|:\s*)", text)[0]
    text = text.strip("-:•. ")
    # Title-case long names
    if 2 <= len(text.split()) <= 8:
        return text
    return None

def _maybe_refine_item_names_with_gemini(data: dict, image_bytes: bytes, mime_type: str):
    """If too many items have placeholder names (e.g., 'Item 1'), ask Gemini for just real dish names and reassign.

    Strategy:
    - Count items whose name matches /^Item\s+\d+$/i.
    - If more than half of all items are placeholders, call Gemini again with a stricter prompt asking ONLY for
      JSON array of dish names in image order, with optional prices.
    - Then reassign names by zipping over the flattened items in order.
    """
    try:
        if not isinstance(data, dict):
            return data
        categories = data.get("categories") or {}
        # Flatten items
        flat_items = []
        cat_keys = list(categories.keys()) if isinstance(categories, dict) else []
        for ck in cat_keys:
            items = categories.get(ck) or []
            for it in items:
                if isinstance(it, dict):
                    flat_items.append(it)
                else:
                    flat_items.append({"name": str(it)})
        if not flat_items:
            return data
        import re as _re
        placeholders = sum(1 for it in flat_items if isinstance(it, dict) and isinstance(it.get("name"), str) and _re.match(r"^Item\s*\d+$", it.get("name").strip(), flags=_re.I))
        if placeholders <= len(flat_items) // 2:
            return data

        # Ask Gemini for just names (and optional prices)
        refine_prompt = (
            "Return ONLY JSON array with the real dish names (and optional prices) in the order they appear in the image. "
            "Structure: [{\"name\": string, \"price\": string (optional)}]. "
            "Never output placeholder names like 'Item 1'."
        )
        refine_parts = [
            refine_prompt,
            {"mime_type": mime_type, "data": image_bytes},
        ]
        refine_cfg = genai.GenerationConfig(response_mime_type="application/json", temperature=0.1)
        ref = client.generate_content(refine_parts, generation_config=refine_cfg)
        arr_txt = (ref.text or "").strip()
        try:
            name_list = json.loads(arr_txt)
        except Exception:
            js = arr_txt[arr_txt.find("[") : arr_txt.rfind("]") + 1]
            name_list = json.loads(js) if js else []
        # Ensure list of dicts with name
        cleaned = []
        for x in name_list:
            if isinstance(x, str):
                cleaned.append({"name": x})
            elif isinstance(x, dict) and x.get("name"):
                cleaned.append({"name": str(x.get("name")), "price": x.get("price")})
        if not cleaned:
            return data
        # Reassign sequentially
        idx = 0
        for ck in cat_keys:
            items = categories.get(ck) or []
            for it in items:
                if idx < len(cleaned) and isinstance(it, dict):
                    if not it.get("name") or _re.match(r"^Item\s*\d+$", str(it.get("name")).strip(), flags=_re.I):
                        it["name"] = cleaned[idx].get("name")
                        # if missing price, adopt refined price
                        if not it.get("price") and cleaned[idx].get("price"):
                            it["price"] = cleaned[idx].get("price")
                    idx += 1
                elif idx < len(cleaned):
                    idx += 1
        return data
    except Exception:
        return data

def _render_results(menu_data, image_style, layout_style, include_nutrition):
    st.markdown("## 🖼️ Your Visual Menu Results")
    st.markdown("### 🍽️ Generated Food Photos with Nutritional Information")
    dish_images = st.session_state.dish_images
    nutritional_data = st.session_state.nutritional_data
    for category, items in menu_data.get('categories', {}).items():
        st.markdown(f"**{category}:**")
        cols = st.columns(min(len(items), 3) or 1)
        for i, item in enumerate(items):
            dish_name = _item_name(item, i)
            col_index = i % max(len(cols), 1)
            with cols[col_index]:
                if dish_name in dish_images:
                    st.image(
                        dish_images[dish_name]['url'],
                        caption=f"{dish_name} - {item.get('price', '')}",
                        width='stretch'
                    )
                    if include_nutrition and dish_name in nutritional_data:
                        nutrition = nutritional_data[dish_name]
                        st.markdown(f"**Nutrition:** {nutrition.get('calories', 'N/A')} cal, {nutrition.get('protein', 'N/A')} protein")
                desc = _item_field(item, 'description', '')
                if desc:
                    st.markdown(f"*{desc}*")

    st.markdown("### 📥 Download Individual Food Photos")
    download_cols = st.columns(3)
    col_count = 0
    timestamp = st.session_state.generation_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    for category, items in menu_data.get('categories', {}).items():
        st.markdown(f"**Download {category} Photos:**")
        for i2, item in enumerate(items):
            dish_name = _item_name(item, i2)
            if dish_name in st.session_state.dish_images:
                clean_name = re.sub(r'[^a-zA-Z0-9]', '_', dish_name)
                with download_cols[col_count % 3]:
                    st.download_button(
                        label=f"💾 {dish_name}",
                        data=st.session_state.dish_images[dish_name]['bytes'],
                        file_name=f"{clean_name}_{timestamp}.png",
                        mime="image/png",
                        width='stretch',
                        key=f"dl-{category}-{i2}-{dish_name}-{timestamp}"
                    )
                col_count += 1

    restaurant_name_clean = re.sub(r'[^a-zA-Z0-9]', '_', menu_data.get('restaurant_name', 'menu'))
    st.markdown("### 📥 Download Menu Report")
    col_download1, col_download2 = st.columns(2)
    with col_download1:
        st.markdown("**📁 Bulk Download:**")
        st.info("Individual photos available above, or use menu report for complete details")
    with col_download2:
        menu_summary = f"""VISUAL MENU GENERATION REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Restaurant: {menu_data.get('restaurant_name', 'N/A')}
Cuisine Type: {menu_data.get('cuisine_type', 'N/A')}
Image Style: {image_style}
Food Description Display: {layout_style}
Nutritional Info: {'Included' if include_nutrition else 'Not Included'}

Menu Items Generated:
"""
        for category, items in menu_data.get('categories', {}).items():
            menu_summary += f"\n{category}:\n"
            for idx3, item in enumerate(items):
                dish_name = _item_name(item, idx3)
                price = _item_field(item, 'price', 'N/A')
                menu_summary += f"- {dish_name} - {price}\n"
                if include_nutrition and dish_name in st.session_state.nutritional_data:
                    nutrition = st.session_state.nutritional_data[dish_name]
                    menu_summary += f"  Nutrition: {nutrition.get('calories', 'N/A')} cal, {nutrition.get('protein', 'N/A')} protein\n"
        menu_summary += f"""

Generation Statistics:
- Total Items: {sum(len(items) for items in menu_data.get('categories', {}).values())}
- Categories: {len(menu_data.get('categories', {}))}
- Food Photos Generated: {len(st.session_state.dish_images)}
- Processing Time: Several minutes

Files Included:
- Individual food photographs for each menu item
- Nutritional information (if selected)

Usage Recommendations:
- Use individual photos for online menus and delivery apps
- Share on social media to showcase specific dishes
- Use for promotional materials and advertisements
- Update photos seasonally or when menu changes
"""
        st.download_button(
            label="📄 Download Menu Report",
            data=menu_summary,
            file_name=f"menu_report_{restaurant_name_clean}_{timestamp}.txt",
            mime="text/plain",
            width='stretch',
            key=f"report-{timestamp}"
        )
def generate_food_image(dish_name, description, cuisine_type, image_style):
    """Generate food image for a specific dish.

    Uses a free prompt-to-image endpoint to avoid requiring an OpenAI or other paid image API key.
    """

    # Style configurations
    style_configs = {
        "Professional Food Photography": {
            "lighting": "professional studio lighting with soft shadows",
            "background": "clean white or neutral background",
            "composition": "centered plating with professional garnish",
            "quality": "restaurant-quality presentation, high-end food photography"
        },
        "Rustic Homestyle": {
            "lighting": "warm, natural lighting",
            "background": "rustic wooden table or textured surface",
            "composition": "casual, homestyle presentation",
            "quality": "comforting, homemade appearance with natural styling"
        },
        "Modern Minimalist": {
            "lighting": "clean, bright lighting",
            "background": "minimalist white or light gray background",
            "composition": "artistic plating with negative space",
            "quality": "contemporary, Instagram-worthy presentation"
        },
        "Vibrant Colorful": {
            "lighting": "bright, vibrant lighting that enhances colors",
            "background": "colorful or complementary background",
            "composition": "dynamic, eye-catching presentation",
            "quality": "bold, appetizing colors that pop"
        }
    }

    style_config = style_configs.get(image_style, style_configs["Professional Food Photography"])

    # Create detailed food photography prompt
    food_prompt = f"""
Professional food photography of {dish_name} ({cuisine_type} cuisine).

Dish details: {description}

Photography specifications:
- {style_config['lighting']}
- {style_config['background']}
- {style_config['composition']}
- {style_config['quality']}

Technical requirements:
- High resolution, commercial food photography quality
- Appetizing and mouth-watering presentation
- Perfect focus and sharp details
- Colors that enhance appetite appeal
- Professional plating and garnish
- Shot from optimal angle to showcase the dish
- No text or watermarks in the image

Style: Photorealistic, magazine-quality food photography that would be used in high-end restaurant menus or food advertising.
"""

    try:
        # Use a public prompt-to-image endpoint (no key required)
        prompt_encoded = requests.utils.quote(food_prompt)
        url = f"https://image.pollinations.ai/prompt/{prompt_encoded}?width=1024&height=1024"
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200 or not resp.content:
            st.error(f"No image data returned for {dish_name}")
            return None, None
        image_bytes = resp.content
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{image_base64}"
        return image_url, image_bytes
    except Exception as e:
        st.error(f"Failed to generate image for {dish_name}: {str(e)}")
        return None, None


# Main UI
st.title("🍽️ AI Menu Visual Generator")
st.markdown("Transform your text-only menu into a stunning visual menu with food photos and nutritional information!")

# Only show main interface if API key is provided
if client:
    st.markdown("### 🖼️ Upload Your Menu Image")
    st.markdown("Upload a photo of your text-only menu and we'll extract all items automatically!")

    uploaded_file = st.file_uploader(
        "Choose menu image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of your menu with readable text"
    )

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Menu", width=400)

        if st.button("🔍Extract Menu Items", type="primary"):
            with st.spinner("🤖 Analyzing menu image and extracting items..."):
                menu_data = extract_menu_text_from_image(uploaded_file)

                if "error" in menu_data:
                    st.error(f"Error: {menu_data['error']}")
                    if "raw_text" in menu_data:
                        st.text_area("Raw extracted text:", menu_data['raw_text'], height=200)
                else:
                    st.success("✅ Menu items extracted successfully!")
                    st.session_state.menu_data = normalize_menu_data(menu_data)
                    menu_data = st.session_state.menu_data

                    # Display extracted menu structure
                    st.markdown("### 📋 Extracted Menu Structure")

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Restaurant:** {menu_data.get('restaurant_name', 'Not detected')}")
                        st.markdown(f"**Cuisine Type:** {menu_data.get('cuisine_type', 'Not detected')}")

                        for category, items in menu_data.get('categories', {}).items():
                            st.markdown(f"**{category}:**")
                            for idx, item in enumerate(items):
                                name = _item_name(item, idx)
                                price = _item_field(item, 'price', 'N/A')
                                desc = _item_field(item, 'description', '')
                                st.markdown(f"- {name} - {price}")
                                if desc:
                                    st.markdown(f"*{desc}*")
                    
                    with col2:
                        st.markdown("**📊 Menu Statistics:**")
                        total_items = sum(len(items) for items in menu_data.get('categories', {}).values())
                        st.metric("Total Items", total_items)
                        st.metric("Categories", len(menu_data.get('categories', {})))

    # Menu processing and generation section
    if 'menu_data' in st.session_state:
        st.markdown("---")
        st.markdown("### 🎨 Generate Visual Menu")

        # Configuration options
        col1, col2, col3 = st.columns(3)

        with col1:
            image_style = st.selectbox(
                "Food Photo Style",
                [
                    "Professional Food Photography",
                    "Rustic Homestyle",
                    "Modern Minimalist",
                    "Vibrant Colorful"
                ],
                help="Choose the style for food photography"
            )

        with col2:
            layout_style = st.selectbox(
                "Food Description Display",
                [
                    "Show Descriptions",
                    "Hide Descriptions",
                    "Short Descriptions Only"
                ],
                help="Choose how to display food descriptions"
            )

        with col3:
            include_nutrition = st.checkbox(
                "Include Nutritional Info",
                value=True,
                help="Add calories and nutrition information to each dish"
            )

        # Generate visual menu
        if 'dish_images' not in st.session_state:
            st.session_state.dish_images = {}
        if 'nutritional_data' not in st.session_state:
            st.session_state.nutritional_data = {}
        if 'generation_timestamp' not in st.session_state:
            st.session_state.generation_timestamp = None

        if st.button("🚀 Generate Complete Visual Menu", type="primary", use_container_width=True):
            menu_data = st.session_state.menu_data

            with st.spinner(
                    "🎨 Creating your visual menu... This may take several minutes as we generate food photos for each item."):

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                dish_images = {}
                nutritional_data = {}

                # Get all dishes
                all_dishes = []
                for category, items in menu_data.get('categories', {}).items():
                    for item in items:
                        all_dishes.append((category, item))
                
                total_dishes = len(all_dishes)

                # Generate images and nutrition for each dish
                for i, (category, item) in enumerate(all_dishes):
                    dish_name = _item_name(item, i)
                    description = _item_field(item, 'description', '')

                    status_text.text(f"Generating food photo for: {dish_name}")

                    # Generate food image
                    image_url, image_bytes = generate_food_image(
                        dish_name,
                        description,
                        menu_data.get('cuisine_type', 'Fine Dining'),
                        image_style
                    )

                    if image_url:
                        dish_images[dish_name] = {
                            'url': image_url,
                            'bytes': image_bytes
                        }

                    # Get nutritional information
                    if include_nutrition:
                        nutrition = get_nutritional_info(dish_name, menu_data.get('cuisine_type', 'Fine Dining'))
                        nutritional_data[dish_name] = nutrition

                    # Update progress
                    progress_bar.progress((i + 1) / total_dishes)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Visual menu generation complete!")

                # Persist results in session_state so they survive reruns
                st.session_state.dish_images = dish_images
                st.session_state.nutritional_data = nutritional_data
                st.session_state.generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if st.session_state.dish_images:
            _render_results(st.session_state.menu_data, image_style, layout_style, include_nutrition)

else:
    st.info("👋 Please enter your Gemini API key above to get started!")

    # Showcase features when no API key
    st.markdown("""
    ## 🚀 Transform Your Menu with AI

    ### 📸 **Upload Any Menu Photo**
    - Take a picture of your text-only menu
    - AI automatically extracts all items, prices, and descriptions
    - Works with handwritten or printed menus

    ### 🍽️ **Generate Professional Food Photos**
    - AI creates appetizing photos for every dish
    - Multiple photography styles available
    - Professional restaurant-quality images

    ### 📊 **Add Nutritional Information**
    - Automatic calorie and nutrition estimates
    - Protein, carbs, fat, and fiber information
    - Helps meet health disclosure requirements
    """)