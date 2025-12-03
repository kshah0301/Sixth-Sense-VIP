#!/bin/bash
CATALOG="data/off_us.jsonl"
SCRIPT="match_item.py"

echo "=== MASS TEST RUN START $(date) ==="

run() {
    echo -e "\n=============================="
    echo "TEST: $*"
    echo "=============================="
    python "$SCRIPT" --catalog "$CATALOG" $*
}

###########################################
# DAIRY
###########################################

run --brand "Horizon Organic" --product "Whole Milk"
run --brand "Organic Valley" --product "2% Milk"
run --brand "Chobani" --product "Greek Yogurt Plain"
run --brand "Fage" --product "Total 5% Yogurt"
run --brand "Tillamook" --product "Medium Cheddar"
run --brand "Sargento" --product "Shredded Cheddar"
run --brand "Philadelphia" --product "Cream Cheese"
run --brand "Kerrygold" --product "Salted Butter"
run --brand "Cabot" --product "Sharp Cheddar"
run --brand "Stonyfield" --product "Vanilla Yogurt"
run --brand "Fairlife" --product "Chocolate Milk"
run --brand "Land O Lakes" --product "Half and Half"
run --brand "Chobani" --product "Oat Milk"
run --brand "Silk" --product "Almond Milk Unsweetened"
run --brand "Siggi's" --product "Skyr Yogurt"
run --brand "Lifeway" --product "Plain Kefir"
run --brand "Daisy" --product "Sour Cream"
run --brand "Reddi Wip" --product "Whipped Cream"
run --brand "Yoplait" --product "Original Strawberry Yogurt"
run --brand "Activia" --product "Probiotic Yogurt"

###########################################
# BEVERAGES
###########################################
run --brand "Coca-Cola" --product "Coke Zero"
run --brand "Pepsi" --product "Diet Pepsi"
run --brand "Gatorade" --product "Fruit Punch"
run --brand "Monster" --product "Energy Drink"
run --brand "Red Bull" --product "Energy Drink"
run --brand "Tropicana" --product "Orange Juice"
run --brand "Simply" --product "Lemonade"
run --brand "Arizona" --product "Iced Tea"
run --brand "Starbucks" --product "Cold Brew"
run --brand "LaCroix" --product "Lime Sparkling Water"
run --brand "Poland Spring" --product "Spring Water"
run --brand "Fiji" --product "Natural Artesian Water"
run --brand "Minute Maid" --product "Apple Juice"
run --brand "Nesquik" --product "Chocolate Milk Drink"
run --brand "Honest Tea" --product "Honey Green Tea"
run --brand "Snapple" --product "Peach Tea"
run --brand "Vita Coco" --product "Coconut Water"
run --brand "Ocean Spray" --product "Cranberry Juice"

###########################################
# PRODUCE
###########################################
run --brand "NatureSweet" --product "Cherry Tomatoes"
run --brand "Driscoll's" --product "Blueberries"
run --brand "Fresh Express" --product "Baby Spinach"
run --brand "Dole" --product "Bananas"
run --brand "Organic Girl" --product "Spring Mix"
run --brand "Grimmway" --product "Baby Carrots"
run --brand "Taylor Farms" --product "Chopped Salad Kit"
run --brand "Envy" --product "Apples"
run --brand "Sunkist" --product "Oranges"
run --brand "Vidalia" --product "Sweet Onions"
run --brand "Russet" --product "Potatoes"
run --brand "Del Monte" --product "Pineapple Chunks"
run --brand "Little Potato Company" --product "Creamer Potatoes"
run --brand "Melissa's" --product "Jalapenos"
run --brand "Cherubs" --product "Grape Tomatoes"

###########################################
# SNACKS
###########################################
run --brand "Lays" --product "Classic Potato Chips"
run --brand "Doritos" --product "Nacho Cheese"
run --brand "Cheez-It" --product "Original"
run --brand "Ritz" --product "Crackers"
run --brand "Oreos" --product "Chocolate Sandwich Cookies"
run --brand "Clif" --product "Chocolate Chip Bar"
run --brand "Nature Valley" --product "Oats n Honey Bar"
run --brand "Goldfish" --product "Cheddar Crackers"
run --brand "Quaker" --product "Rice Cakes"
run --brand "Kind" --product "Peanut Butter Bar"
run --brand "Welch's" --product "Fruit Snacks"
run --brand "Pretzel Crisps" --product "Original Pretzel"
run --brand "Popcorners" --product "Kettle Corn"
run --brand "Smartfood" --product "White Cheddar Popcorn"
run --brand "Tostitos" --product "Scoops Tortilla Chips"
run --brand "Pop-Tarts" --product "Strawberry"

###########################################
# PANTRY
###########################################
run --brand "Barilla" --product "Spaghetti"
run --brand "Classico" --product "Tomato Sauce"
run --brand "Heinz" --product "Ketchup"
run --brand "Hunt's" --product "Diced Tomatoes"
run --brand "Bush's" --product "Black Beans"
run --brand "Uncle Ben's" --product "Long Grain Rice"
run --brand "Lundberg" --product "Jasmine Rice"
run --brand "Skippy" --product "Peanut Butter"
run --brand "Nutella" --product "Hazelnut Spread"
run --brand "Jif" --product "Creamy Peanut Butter"
run --brand "Quaker" --product "Oats"
run --brand "Swanson" --product "Chicken Broth"
run --brand "Annie's" --product "Mac and Cheese"
run --brand "Kikkoman" --product "Soy Sauce"
run --brand "Bertolli" --product "Olive Oil"
run --brand "Blue Diamond" --product "Almonds"
run --brand "McCormick" --product "Black Pepper"
run --brand "King Arthur" --product "All Purpose Flour"
run --brand "Domino" --product "Granulated Sugar"
run --brand "Bragg" --product "Apple Cider Vinegar"

###########################################
# FROZEN
###########################################
run --brand "Eggo" --product "Homestyle Waffles"
run --brand "Amy's" --product "Macaroni and Cheese"
run --brand "Red Baron" --product "Frozen Pizza"
run --brand "Totino's" --product "Pizza Rolls"
run --brand "Birds Eye" --product "Frozen Peas"
run --brand "Green Giant" --product "Broccoli Florets"
run --brand "Stouffer's" --product "Lasagna"
run --brand "Ben & Jerry's" --product "Chocolate Fudge Brownie"
run --brand "Haagen-Dazs" --product "Vanilla Ice Cream"
run --brand "Talenti" --product "Sea Salt Caramel Gelato"
run --brand "Lean Cuisine" --product "Cheese Pizza"
run --brand "Ore-Ida" --product "French Fries"
run --brand "Smart Ones" --product "Turkey Wrap"

echo "=== MASS TEST RUN END $(date) ==="
