#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

int main() {
    bool success = true;
    const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> normalization = {
        {{}, {}},
        {{0x65, 0x301}, {0xE9}},
        {{0xE9}, {0xE9}},
        {{0x212B}, {0xC5}},
        {{0x61, 0x315, 0x300}, {0xE0, 0x315}},
        {{0x61, 0x305, 0x301}, {0x61, 0x305, 0x301}},
        {{0x301, 0x327}, {0x327, 0x301}},
        {{0x1100, 0x1161, 0x11A8}, {0xAC01}},
        {{0xAC01}, {0xAC01}},
        {{0xFB01, 0xFF21}, {0xFB01, 0xFF21}},
        {{0x958}, {0x915, 0x93C}},
        {{0x1D15E}, {0x1D157, 0x1D165}},
    };
    for (const auto & test : normalization) {
        if (unicode_cpts_normalize_nfc(test.first) != test.second || unicode_cpts_normalize_nfc(test.second) != test.second) {
            fprintf(stderr, "unexpected NFC normalization\n");
            success = false;
            break;
        }
    }

    const struct {
        std::string text;
        std::vector<uint32_t> expected;
    } utf8_tests[] = {
        { "\x7f",                 { 0x7f } },
        { "\xc2\x80",             { 0x80 } },
        { "\xdf\xbf",             { 0x7ff } },
        { "\xe0\xa0\x80",         { 0x800 } },
        { "\xef\xbf\xbf",         { 0xffff } },
        { "\xf0\x90\x80\x80",     { 0x10000 } },
        { "\xf4\x8f\xbf\xbf",     { 0x10ffff } },
        // Invalid-input cases preserve the existing recovery behavior: one U+FFFD per byte.
        { "\xc0\xa0",             { 0xfffd, 0xfffd } },
        { "\xc1\xbf",             { 0xfffd, 0xfffd } },
        { "\xe0\x80\xa0",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xe0\x9f\xbf",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x80\x80\xa0",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x8f\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf4\x90\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf5\x80\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf7\xbf\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xe2\x80",             { 0xfffd, 0xfffd } },
    };

    for (const auto & test : utf8_tests) {
        auto expected = test.expected;
        expected.insert(expected.begin(), 'a');
        expected.push_back('b');
        if (unicode_cpts_from_utf8("a" + test.text + "b") != expected) {
            fprintf(stderr, "unexpected UTF-8 decoding for bytes:");
            for (const unsigned char byte : test.text) {
                fprintf(stderr, " %02x", byte);
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    const std::vector<std::string> regex_exprs = {
        "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
    };
    const std::vector<std::string> expected = { " ~", "foo" };
    const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "unexpected split:");
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        success = false;
    }

    // LLAMA_VOCAB_PRE_TYPE_BAILINGMOE (Ling), the expected splits come from the pre-tokenizer of the Ling-3.2-air HF tokenizer
    const std::vector<std::string> regex_bailingmoe = {
        "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
    };
    const struct {
        std::string text;
        std::vector<std::string> expected;
    } bailingmoe_tests[] = {
        { "Hello world", { "Hello", " world" } },
        { " Hello  world ", { " Hello", " ", " world", " " } },
        { "I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL", { "I", "'ve", " been", " '", "told", " he", "'s", " there", ",", " '", "RE", " you", " sure", "?", " '", "M", " not", " sure", " I", "'ll", " make", " it", ",", " '", "D", " you", " like", " some", " tea", "?", " We", "'Ve", " a", "'lL" } },
        { "3 33 333 3333 3.14 1,000,000", { "3", " ", "3", "3", " ", "3", "3", "3", " ", "3", "3", "3", "3", " ", "3", ".", "1", "4", " ", "1", ",", "0", "0", "0", ",", "0", "0", "0" } },
        { "line one\nline two\r\n\n  indented\ttab", { "line", " one", "\n", "line", " two", "\r\n\n", " ", " indented", "\ttab" } },
        { "foo!!! bar??? (baz) [qux] {x}", { "foo", "!!!", " bar", "???", " (", "baz", ")", " [", "qux", "]", " {", "x", "}" } },
        { "\346\210\221\346\203\263\345\234\250apple\345\267\245\344\275\2341314151\345\244\251\357\275\236", { "\346\210\221\346\203\263\345\234\250apple\345\267\245\344\275\234", "1", "3", "1", "4", "1", "5", "1", "\345\244\251", "\357\275\236" } },
        { "\320\275\320\265\321\211\320\276 \320\275\320\260 \320\221\321\212\320\273\320\263\320\260\321\200\321\201\320\272\320\270", { "\320\275\320\265\321\211\320\276", " \320\275\320\260", " \320\221\321\212\320\273\320\263\320\260\321\200\321\201\320\272\320\270" } },
        { "\360\237\232\200 (normal) \360\237\230\266\342\200\215\360\237\214\253\357\270\217 \342\234\205 \360\237\246\231\360\237\246\231", { "\360\237\232\200", " (", "normal", ")", " \360\237\230\266\342\200\215\360\237\214\253\357\270\217", " \342\234\205", " \360\237\246\231\360\237\246\231" } },
        { "a-b_c.d/e\\f", { "a", "-b", "_c", ".d", "/e", "\\f" } },
        { "   \n\n   x", { "   \n\n", "  ", " x" } },
        { "def f(x):\n    return x**2  # comment\n", { "def", " f", "(x", "):\n", "   ", " return", " x", "**", "2", " ", " #", " comment", "\n" } },
        { "<think>ok</think><tool_call>", { "<think", ">ok", "</", "think", "><", "tool", "_call", ">" } },
        { "\303\204\303\226\303\234 \303\244\303\266\303\274 \303\237 caf\303\251", { "\303\204\303\226\303\234", " \303\244\303\266\303\274", " \303\237", " caf\303\251" } },
        { "...\n\n---\n", { "...\n\n", "---\n" } },
    };
    for (const auto & test : bailingmoe_tests) {
        const auto pieces = unicode_regex_split(test.text, regex_bailingmoe, false);
        if (pieces != test.expected) {
            fprintf(stderr, "unexpected bailingmoe split of \"%s\":", test.text.c_str());
            for (const auto & piece : pieces) {
                fprintf(stderr, " [%s]", piece.c_str());
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    // LLAMA_VOCAB_PRE_TYPE_GPT4O has a custom splitter, wrapping the regex in a group makes it use the std::regex fallback instead
    const std::string regex_gpt4o = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const std::string gpt4o_tests[] = {
        "Hello World HELLO wORLD helloWorld HTTPServer xmlHTTPRequest",
        "I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL A'S",
        "3 33 333 3333 3.14 1,000,000 \331\241\331\242\331\243\331\244 \342\205\247\302\262",
        "line one\nline two\r\n\n  indented\ttab   \n\n   x  \v\f end  ",
        "foo!!! bar??? (baz) [qux] {x} a/b//\n c// \r\n -- \342\200\224 \302\277qu\303\251?",
        "\346\210\221\346\203\263\345\234\250apple\345\267\245\344\275\234 \320\221\321\212\320\273\320\263\320\260\321\200\321\201\320\272\320\270 \303\204\303\226\303\234 caf\303\251 cafe\314\201 Cafe\314\201S \314\201x .\314\201a",
        "\360\237\232\200 (normal) \360\237\230\266\342\200\215\360\237\214\253\357\270\217 \342\202\254100 \302\240x\343\200\200y\342\200\250z\302\205w \x01\x7f\xff",
        "def f(x):\n    return x**2  # comment\n<think>ok</think><tool_call>\"k\": [1, 2]}",
    };
    for (const auto & text : gpt4o_tests) {
        const auto pieces = unicode_regex_split(text, { regex_gpt4o }, false);
        if (pieces != unicode_regex_split(text, { "(?:" + regex_gpt4o + ")" }, false)) {
            fprintf(stderr, "unexpected gpt4o split of \"%s\":", text.c_str());
            for (const auto & piece : pieces) {
                fprintf(stderr, " [%s]", piece.c_str());
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    // LLAMA_VOCAB_PRE_TYPE_QWEN2 (Kolibri), the expected splits come from the pre-tokenizer of the Kolibri-1.1 HF tokenizer
    const std::vector<std::string> regex_qwen2 = {
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    const struct {
        std::string text;
        std::vector<std::string> expected;
    } qwen2_tests[] = {
        { "Hello world", { "Hello", " world" } },
        { " Hallo  Welt ", { " Hallo", " ", " Welt", " " } },
        { "Gr\303\274\303\237e aus M\303\274nchen, \303\204rger \303\274ber \303\226l und Stra\303\237en\303\244cker", { "Gr\303\274\303\237e", " aus", " M\303\274nchen", ",", " \303\204rger", " \303\274ber", " \303\226l", " und", " Stra\303\237en\303\244cker" } },
        { "I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL", { "I", "'ve", " been", " '", "told", " he", "'s", " there", ",", " '", "RE", " you", " sure", "?", " '", "M", " not", " sure", " I", "'ll", " make", " it", ",", " '", "D", " you", " like", " some", " tea", "?", " We", "'Ve", " a", "'lL" } },
        { "3 33 333 3333 3.14 1.000.000,50\342\202\254", { "3", " ", "3", "3", " ", "3", "3", "3", " ", "3", "3", "3", "3", " ", "3", ".", "1", "4", " ", "1", ".", "0", "0", "0", ".", "0", "0", "0", ",", "5", "0", "\342\202\254" } },
        { "Die Bundesrepublik Deutschland (BRD) hat 16 L\303\244nder.", { "Die", " Bundesrepublik", " Deutschland", " (", "BRD", ")", " hat", " ", "1", "6", " L\303\244nder", "." } },
        { "line one\nline two\r\n\n  indented\ttab", { "line", " one", "\n", "line", " two", "\r\n\n", " ", " indented", "\ttab" } },
        { "foo!!! bar??? (baz) [qux] {x}", { "foo", "!!!", " bar", "???", " (", "baz", ")", " [", "qux", "]", " {", "x", "}" } },
        { "\342\200\236Anf\303\274hrungszeichen\342\200\234 und \302\273Guillemets\302\253 \342\200\223 Gedankenstrich", { "\342\200\236Anf\303\274hrungszeichen", "\342\200\234", " und", " \302\273", "Guillemets", "\302\253", " \342\200\223", " Gedankenstrich" } },
        { "STRASSE \341\272\236 gro\303\237", { "STRASSE", " \341\272\236", " gro\303\237" } },
        { "   \n\n   x", { "   \n\n", "  ", " x" } },
        { "def f(x):\n    return x**2  # comment\n", { "def", " f", "(x", "):\n", "   ", " return", " x", "**", "2", " ", " #", " comment", "\n" } },
        { "<think>ok</think><tool_call>", { "<think", ">ok", "</", "think", "><", "tool", "_call", ">" } },
        { "<|im_start|>assistant\n", { "<|", "im", "_start", "|>", "assistant", "\n" } },
        { "...\n\n---\n", { "...\n\n", "---\n" } },
    };
    for (const auto & test : qwen2_tests) {
        const auto pieces = unicode_regex_split(test.text, regex_qwen2, false);
        if (pieces != test.expected) {
            fprintf(stderr, "unexpected qwen2 split of \"%s\":", test.text.c_str());
            for (const auto & piece : pieces) {
                fprintf(stderr, " [%s]", piece.c_str());
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    return success ? 0 : 1;
}
