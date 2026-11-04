//! Built-in calculator: exact answers to arithmetic and unit-conversion questions.
//!
//! The 2.9B chat model answers maths fluently but not reliably -- tested 2026-11-04 it said
//! 5 yards = 1.524 m (that is 5 *feet*) and 3.5 m = 11.81 ft (it is 11.48), with the same
//! confidence as its right answers. Questions that are clearly maths are therefore rewritten into
//! calculator syntax and evaluated by fend-core (MIT, no dependencies, unit-aware, arbitrary
//! precision), the same way `try_answer_time_date_deterministically` answers clock questions.
//!
//! Chosen over numbat after testing both: numbat read "5 yards in meters" as "in" = *inches* and
//! answered "0.151891 yd³" without complaint; fend understands "in"/"to"/"of"/"%" as people use
//! them and returns an error when it doesn't understand, never a silently wrong number.
//!
//! Anything that isn't clearly maths, or that fend can't evaluate, returns `None` and goes to the
//! chat model exactly as before -- this can only make answers more exact, never block one.

/// Phrases stripped from the start of a question before evaluation.
const LEAD_INS: [&str; 16] = [
    "please ", "can you ", "could you ", "tell me ", "what is ", "what's ", "whats ", "how much is ",
    "calculate ", "compute ", "convert ", "work out ", "figure out ", "solve ", "evaluate ", "the ",
];

/// Phrases stripped from the end (the quiz asked "... Just the number.").
const TAIL_OFFS: [&str; 6] = ["just the number", "only the number", "exactly", "please", "for me", "approximately"];

/// Rewrites a question into fend syntax, or None if it doesn't look like a maths question.
pub fn to_expression(question: &str) -> Option<String> {
    let mut s = question.trim().to_lowercase();
    // "how many seconds in a day" -> "... in 1 day": "a"/"an" is the number here.
    if s.starts_with("how many ") {
        s = s.replacen(" in a ", " in 1 ", 1).replacen(" in an ", " in 1 ", 1);
    }
    if !s.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    // Keep only the first sentence that has a digit in it ("What is 6 times 7? Just the number.").
    if let Some(sentence) = s
        .split_inclusive(['?', '!'])
        .find(|part| part.chars().any(|c| c.is_ascii_digit()))
    {
        s = sentence.to_string();
    }
    s = s.trim().trim_end_matches(['?', '.', '!', ' ']).to_string();
    for tail in TAIL_OFFS {
        if let Some(rest) = s.strip_suffix(tail) {
            s = rest.trim_end_matches([',', '.', ' ']).to_string();
        }
    }
    // "how many meters are in 5 yards" -> "5 yards to meters"
    if let Some(rest) = s.strip_prefix("how many ") {
        for sep in [" are there in ", " are in ", " is in ", " in "] {
            if let Some((target, amount)) = rest.split_once(sep) {
                s = format!("{} to {}", amount.trim(), target.trim());
                break;
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for lead in LEAD_INS {
            if let Some(rest) = s.strip_prefix(lead) {
                s = rest.to_string();
                changed = true;
            }
        }
    }

    // Words people use for operators and functions.
    let replacements: &[(&str, &str)] = &[
        (" as a percentage", " to %"),
        (" as a percent", " to %"),
        (" as percentage", " to %"),
        (" as percent", " to %"),
        (" in percent", " to %"),
        ("square root of ", "sqrt "),
        ("cube root of ", "cbrt "),
        (" multiplied by ", " * "),
        (" times ", " * "),
        (" divided by ", " / "),
        (" plus ", " + "),
        (" minus ", " - "),
        (" squared", "^2"),
        (" cubed", "^3"),
        (" to the power of ", "^"),
        (" percent of ", "% of "),
        ("degrees fahrenheit", "°F"),
        ("degrees celsius", "°C"),
        ("degrees centigrade", "°C"),
        ("fahrenheit", "°F"),
        ("celsius", "°C"),
    ];
    for (from, to) in replacements {
        s = s.replace(from, to);
    }
    // "98.6 F to C" -> "98.6 °F to °C"
    s = temperature_letters(&s);
    // "6 x 7" -> "6 * 7" (only between numbers; a lone x is otherwise left alone)
    s = replace_x_between_numbers(&s);
    // "half of 90" -> "0.5 * 90"
    s = s.replace("half of ", "0.5 * ").replace("a quarter of ", "0.25 * ");

    // Must now look like a calculation: an operator, a function, or a conversion.
    let has_op = s.contains(['+', '*', '/', '^', '%'])
        || s.contains(" - ")
        || s.contains("sqrt")
        || s.contains("cbrt")
        || s.contains(" to ")
        || s.contains(" in ");
    if !has_op {
        return None;
    }
    // Long sentences are word problems or chat, not calculator input.
    if s.split_whitespace().count() > 10 {
        return None;
    }
    // Words that mean the question is about something else (a story, a count, a list...).
    const NOT_MATHS: [&str; 12] = [
        "story", "write", "count from", "list", "poem", "name", "explain", "why", "who", "when",
        "where", "if ",
    ];
    if NOT_MATHS.iter().any(|w| s.contains(w)) {
        return None;
    }
    Some(s.trim().to_string())
}

/// A bare "f" or "c" is a temperature only right after a number or after "to"/"in"
/// ("98.6 f to c"); anywhere else the letter is left alone.
fn temperature_letters(s: &str) -> String {
    let words: Vec<&str> = s.split(' ').collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    for (i, w) in words.iter().enumerate() {
        let prev = if i > 0 { words[i - 1] } else { "" };
        let after_number = prev.parse::<f64>().is_ok();
        let after_to = prev == "to" || prev == "in";
        out.push(match *w {
            "f" | "°f" if after_number || after_to => "°F".to_string(),
            "c" | "°c" if after_number || after_to => "°C".to_string(),
            other => other.to_string(),
        });
    }
    out.join(" ")
}

/// Numbers in a sentence, split into percentages ("18%", "18 percent") and plain amounts
/// ("$45", "45 dollars", "45"). Commas in "1,200" are thousands separators.
fn amounts(s: &str) -> (Vec<f64>, Vec<f64>) {
    let chars: Vec<char> = s.chars().collect();
    let (mut percents, mut others) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() || (chars[i] == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == ',') {
                i += 1;
            }
            let text: String = chars[start..i].iter().filter(|c| **c != ',').collect();
            let text = text.trim_end_matches('.');
            if let Ok(n) = text.parse::<f64>() {
                let rest: String = chars[i..].iter().collect();
                let rest = rest.trim_start();
                if rest.starts_with('%') || rest.starts_with("percent") {
                    percents.push(n);
                } else {
                    others.push(n);
                }
            }
        } else {
            i += 1;
        }
    }
    (percents, others)
}

/// "$1,234.50"
fn money(x: f64) -> String {
    let s = format!("{:.2}", x.abs());
    let (whole, cents) = s.split_once('.').unwrap_or((&s, "00"));
    let mut grouped = String::new();
    for (i, ch) in whole.chars().enumerate() {
        if i > 0 && (whole.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    format!("{}${grouped}.{cents}", if x < 0.0 { "-" } else { "" })
}

/// "an 18% tip", "an 8% tip", "an 11% tip", but "a 15% tip": the article follows how the number
/// is said out loud.
fn article_for(p: f64) -> &'static str {
    let whole = p.trunc() as i64;
    let digits = whole.to_string();
    let vowel_sound = digits.starts_with('8') || whole == 11 || whole == 18;
    if vowel_sound { "An" } else { "A" }
}

fn pct(x: f64) -> String {
    let t = format!("{x:.2}");
    t.trim_end_matches('0').trim_end_matches('.').to_string() + "%"
}

/// Shopping maths with a money-shaped answer: discounts ("20% off $50") and tips ("18% tip on
/// $45"). Exactly one percentage and one amount, in a short sentence; anything else is left to
/// the general calculator or the model.
fn try_shopping(question: &str) -> Option<String> {
    let s = question.to_lowercase();
    if s.split_whitespace().count() > 14 {
        return None;
    }
    let (percents, others) = amounts(&s);
    if percents.len() != 1 || others.len() != 1 {
        return None;
    }
    let (p, amount) = (percents[0], others[0]);
    let is_word = |w: &str| s.split(|c: char| !c.is_alphanumeric()).any(|x| x == w);
    if is_word("off") || is_word("discount") || is_word("discounted") {
        let saved = amount * p / 100.0;
        return Some(format!(
            "{} off {} is {} (you save {}).",
            pct(p),
            money(amount),
            money(amount - saved),
            money(saved)
        ));
    }
    if is_word("tip") || is_word("gratuity") {
        let tip = amount * p / 100.0;
        return Some(format!(
            "{} {} tip on {} is {}, so the total is {}.",
            article_for(p),
            pct(p),
            money(amount),
            money(tip),
            money(amount + tip)
        ));
    }
    None
}

fn replace_x_between_numbers(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        let prev_digit = i >= 2 && chars[i - 1] == ' ' && chars[i - 2].is_ascii_digit();
        let next_digit = i + 2 < chars.len() && chars[i + 1] == ' ' && chars[i + 2].is_ascii_digit();
        if (c == 'x' || c == '×') && prev_digit && next_digit {
            out.push('*');
        } else {
            out.push(c);
        }
    }
    out
}

/// Shortens fend's full-precision numbers ("approx. 11.4829396325 feet") to something readable
/// ("about 11.483 feet"), keeping exact results exact.
fn tidy_result(raw: &str) -> String {
    let (approx, body) = match raw.strip_prefix("approx. ") {
        Some(rest) => (true, rest),
        None => (false, raw),
    };
    let num_end = body
        .char_indices()
        .find(|(_, c)| !(c.is_ascii_digit() || *c == '.' || *c == '-' || *c == ','))
        .map(|(i, _)| i)
        .unwrap_or(body.len());
    let (num, unit) = body.split_at(num_end);
    // Round when fend says it's approximate, or when an exact result has more decimals than
    // anyone wants read out ("42.1648128 kilometers").
    let long_decimals = num.split_once('.').is_some_and(|(_, d)| d.len() > 4);
    let approx = approx || long_decimals;
    let shown = match (approx, num.replace(',', "").parse::<f64>()) {
        (true, Ok(n)) => {
            let decimals = if n.abs() >= 100.0 { 2 } else if n.abs() >= 1.0 { 3 } else { 5 };
            let mut t = format!("{n:.decimals$}");
            if t.contains('.') {
                t = t.trim_end_matches('0').trim_end_matches('.').to_string();
            }
            t
        }
        _ => num.to_string(),
    };
    format!("{}{}{}", if approx { "about " } else { "" }, shown, tidy_unit(unit))
}

/// "km / h" -> "km/h", "feet^2" -> "square feet", "meters^3" -> "cubic meters".
fn tidy_unit(unit: &str) -> String {
    let unit = unit.replace(" / ", "/");
    let base = unit.trim();
    for (suffix, word) in [("^2", "square"), ("^3", "cubic")] {
        if let Some(name) = base.strip_suffix(suffix) {
            if !name.is_empty() && !name.contains(['^', ' ', '/', '*']) {
                return format!(" {word} {name}");
            }
        }
    }
    unit.replace("^2", "²").replace("^3", "³")
}

/// Readable form of the question side: "5 yards to meters" -> "5 yards", "17 * 23" -> "17 × 23".
fn tidy_question(expr: &str) -> String {
    let left = expr.split(" to ").next().unwrap_or(expr);
    let left = left.split(" in ").next().unwrap_or(left);
    // A spaced "/" is division ("1000 / 8" -> "1000 ÷ 8"); an unspaced one is a fraction ("3/4").
    left.replace('*', "×").replace(" / ", " ÷ ").replace("sqrt ", "√").replace("^2", "²").replace("^3", "³")
}

/// Exact answer for a maths question, or None to let the chat model answer.
pub fn try_answer(question: &str) -> Option<String> {
    if let Some(answer) = try_shopping(question) {
        return Some(answer);
    }
    let expr = to_expression(question)?;
    let mut ctx = fend_core::Context::new();
    let result = fend_core::evaluate(&expr, &mut ctx).ok()?;
    let raw = result.get_main_result().trim().to_string();
    // A usable answer starts with a number (fend can also return functions, dates, text...).
    let starts_numeric = raw.trim_start_matches("approx. ").starts_with(|c: char| c.is_ascii_digit() || c == '-');
    if raw.is_empty() || !starts_numeric {
        return None;
    }
    let answer = tidy_result(&raw);
    let is_conversion = expr.contains(" to ") || expr.contains(" in ");
    Some(if is_conversion {
        format!("{} is {}.", tidy_question(&expr), answer)
    } else {
        format!("{} = {}.", tidy_question(&expr), answer)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(q: &str) -> String {
        try_answer(q).unwrap_or_else(|| panic!("no calculator answer for {q:?} (expr {:?})", to_expression(q)))
    }

    #[test]
    fn the_questions_the_model_got_wrong() {
        assert_eq!(answer("How many meters are in 5 yards?"), "5 yards is 4.572 meters.");
        assert_eq!(answer("How many feet are in 3.5 meters?"), "3.5 meters is about 11.483 feet.");
    }

    #[test]
    fn the_rest_of_the_quiz() {
        assert_eq!(answer("Convert 100 yards to meters."), "100 yards is 91.44 meters.");
        assert_eq!(answer("Convert 26.2 miles to kilometers."), "26.2 miles is about 42.165 kilometers.");
        assert!(answer("What is 72 degrees Fahrenheit in Celsius?").starts_with("72 °F is about 22.222"));
        assert_eq!(answer("What is 15% of 240?"), "15% of 240 = 36.");
        assert_eq!(answer("What is 17 times 23?"), "17 × 23 = 391.");
        assert_eq!(answer("What is the square root of 144?"), "√144 = 12.");
        assert_eq!(answer("Convert 2 pounds to kilograms."), "2 pounds is about 0.90718 kilograms.");
        assert_eq!(answer("What is 6 times 7? Just the number."), "6 × 7 = 42.");
    }

    #[test]
    fn more_conversions_and_arithmetic() {
        assert_eq!(answer("10 km in miles"), "10 km is about 6.214 miles.");
        assert_eq!(answer("what is 6 x 7"), "6 × 7 = 42.");
        assert_eq!(answer("12 squared"), "12² = 144.");
        assert_eq!(answer("what's 1000 divided by 8"), "1000 ÷ 8 = 125.");
        assert_eq!(answer("half of 90"), "0.5 × 90 = 45.");
        assert_eq!(answer("3 cups to ml"), "3 cups is about 709.76 ml.");
        assert_eq!(answer("100 yards to meters"), "100 yards is 91.44 meters.", "short exact results stay exact");
    }

    #[test]
    fn the_gaps_found_in_the_wider_check() {
        assert_eq!(answer("how many seconds in a day"), "1 day is 86400 seconds.");
        assert_eq!(answer("How many cups in a quart?"), "1 quart is 4 cups.");
        assert_eq!(answer("Convert 98.6 F to C"), "98.6 °F is 37 °C.");
        assert_eq!(answer("What is 3/4 as a percentage?"), "3/4 is 75%.");
        assert_eq!(answer("What is 20% off 50 dollars?"), "20% off $50.00 is $40.00 (you save $10.00).");
        assert_eq!(answer("Tip on a $45 bill at 18%?"), "An 18% tip on $45.00 is $8.10, so the total is $53.10.");
        assert_eq!(answer("what's a 15 percent tip on 1,200"), "A 15% tip on $1,200.00 is $180.00, so the total is $1,380.00.");
        assert_eq!(answer("20% tip on $87.50"), "A 20% tip on $87.50 is $17.50, so the total is $105.00.");
        assert_eq!(money(1234567.5), "$1,234,567.50");
        assert_eq!(article_for(8.0), "An");
        assert_eq!(article_for(80.0), "An");
        assert_eq!(article_for(11.0), "An");
        assert_eq!(article_for(12.0), "A");
    }

    #[test]
    fn units_read_naturally() {
        assert_eq!(answer("what is 30 mph in km/h"), "30 mph is about 48.28 km/h.");
        assert_eq!(answer("How many square feet in 2 acres?"), "2 acres is 87120 square feet.");
        assert_eq!(answer("1000 / 8"), "1000 ÷ 8 = 125.");
    }

    #[test]
    fn letters_f_and_c_are_only_temperatures_next_to_numbers() {
        assert_eq!(temperature_letters("98.6 f to c"), "98.6 °F to °C");
        assert_eq!(temperature_letters("plan c for 5 people"), "plan c for 5 people");
        // shopping needs exactly one percentage and one amount
        assert!(try_shopping("20% off 50 dollars or 30% off 80").is_none());
        assert!(try_shopping("I left a tip").is_none());
    }

    #[test]
    fn ordinary_chat_is_left_to_the_model() {
        for q in [
            "Write a 400 word story about a fox who finds a lantern in the woods.",
            "Count from 1 to 10.",
            "What is my name?",
            "I have 3 kids and 2 dogs",
            "Tell me a fun fact about the ocean.",
            "A car travels 150 miles in 2.5 hours. What is its average speed in miles per hour?",
            "Name 5 planets in the solar system",
            "What happened in 1969?",
            "hello",
        ] {
            assert!(try_answer(q).is_none(), "should not be answered by the calculator: {q:?} -> {:?}", try_answer(q));
        }
    }
}
