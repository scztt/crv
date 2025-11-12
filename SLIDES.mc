 
<!-- theme: gradient -->

---

![[Opening slide.png]]

note: 
0:00

---

#  🙃 Okay, this is a pretty broad topic...

**What would it look like to:**
- Use any and all c++20/23/26 features we want
- Emphasize C++ for it's DSL-building skills
- Build our dream set inner-loop DSP abstractions...


---
### Goals

- Explore new C++ features, especially how they work together to express complex semantic concepts
- Re-think the syntax and semantics of optimized DSP code
- Develop a constellation of patterns, and see what happens when we use them together
- Eventually make some sound

---
### Inspirations

- Halide (https://github.com/halide/Halide) "separate implementation from execution"
- Petra (https://github.com/jacquelinekay/petra)

---
### What we'll look at

1. Compile time / runtime variant type, and hoisting
2. Constexpr directed graphs
3. Flexible vectorization
4. Parametric benchmarking and optimization

---

# Part 1: Compile time / runtime variants, and hoisting

---
### What do I mean by hoisting?

```c++
auto amplitude_modulation(auto input, auto freq, auto amp) {
	auto sine_mod = 1 + (sine(freq) * amp);
	return input * sine_mod;
}
```

What happens if amp == 0? What happens if freq is audio rate modulated or constant over time?

note:
3:21

---

Q: How can we **compile different versions of `amplitude_modulation` for each assumption**, and then select the implementation we want (hoisting)?

Q: How can we **pass in the concept "amp == 1"** to this function?

Q: This example is trivial, but if we generalize the concept of hoisting, what else can it cover?

---
### Where to start:  compile time values

(e.g. improved `std::integral_constant`)

```c++
template <typename storage_t, storage_t value>
struct constant_value
{
    using value_t = storage_t;
	static constexpr auto is_constant = true;
	static constexpr auto static_value = value;

    constexpr auto get() const -> auto const&
    {
        return static_value;
    }
        
	constexpr operator value_t() const -> decltype(auto)
    {
        return get();
    }
};

template <auto value>
constexpr auto cv = constant_value<decltype(value), value>;
```

---

### Example

```c++
auto zero = cv<0.0>;

amplitude_modulation(signal, freq, zero);

// later...
auto sine_mod = 1 + (sine(freq) * zero);

amplitude_modulation(signal, freq, amp); // <-- calls a different function...
```


---
### A pleasant surprise...

```c++
auto zero = cv<0.0>;

template <double value>
struct something;

auto instance = something<zero>{} // <--- this works
```

---

### For symmetry, we create a `runtime_value` type with the same interface, but simply wrapping the value.

```c++
template <typename storage_t>
struct runtime_value
{
    using value_t = storage_t;
	static constexpr auto is_constant = true;
	
	auto value;

    constexpr auto get() const -> auto const&
    {
        return value;
    }
    
    constexpr operator value_t() const -> decltype(auto)
    {
        return get();
    }
};
```

---
### Problem: I want to apply a function to my value, but keep the cv / rv-ness of it

**unwrap value → apply function → get result → wrap value**

---
### Solution: We need an applicative

```c++
template <auto func, typename... args_t>
auto apply_with_func(args_t&&... args)
{
    if constexpr (/* function result is either cv or constexpr) */)
    {
        constexpr auto result = func(unwrap_value(fwd(args))...);

        if constexpr (is_cv_any<decltype(result)>)
        {
            return result;
        }
        else
        {
            return constant_value<decltype(result), result>;
        }
    } else if /*...runtime case...*/
}

auto result = apply_with_func<[](auto v) { return std::abs(v); })>(cv<-10>);
static_assert(result.get() == 10);
```

---
### Problem:  All of my cv/rv values are potentially different types - how can I keep some level of strong typing?

---
### A: define concepts to describe our cv types...

```c++
// constant or runtime, any type
template <typename type_t>
concept is_crv_any = requires {
    { std::remove_cvref_t<type_t>::is_constant } -> std::convertible_to<bool>;
};

// constant or runtime, specific type
template <typename type_t, typename value_t>
concept is_crv = requires {
    { std::remove_cvref_t<type_t>::is_constant } -> std::convertible_to<bool>;
    requires std::is_convertible_v<typename std::remove_cvref_t<type_t>::value_t, value_t>;
};

// constant, any type
template <typename type_t>
concept is_cv_any = is_crv_any<type_t> && std::remove_cvref_t<type_t>::is_constant;

// constant, specific type
template <typename type_t, typename value_t>
concept is_cv = is_crv<type_t, value_t> && std::remove_cvref_t<type_t>::is_constant;å

// runtime, any type
template <typename type_t>
concept is_rv_any = is_crv_any<type_t> && !std::remove_cvref_t<type_t>::is_constant;

// runtime, specific type
template <typename type_t, typename value_t>
concept is_rv = is_crv<type_t, value_t> && !std::remove_cvref_t<type_t>::is_constant;
```

---

### concepts + auto → flexible type constraints

```c++
auto amplitude_modulation(auto input, is_crv_any auto freq, is_crv_any auto amp);

auto amplitude_modulation(auto input, is_crv<float> auto freq, is_crv<float> auto amp);
```

---
### Q: What about aggregate data?

```c++
struct values
{
    float freq;
    float amp;
};

auto params = cv<values{ 440.0f, 0.5f }>;
```

But there's a problem: **it's all-constant or nothing.**

---

```c++
template <is_crv<float> freq_t, is_crv<float> amp_t>
struct values
{
    freq_t freq;
    amp_t amp;
};
```

#### **This is okay, but...**
1. Verbose (esp. for nested types)
2. Each member is `size==1` even if its "empty"
3. We want **parametric types**, we want to be able to transform them easily

#### Maybe what we want is **property lists** / key-value pairs...

---


```c++
template <typename... property_specs_t>
struct properties
{
    template <typename... property_instances_t>
    struct instance : property_instances_t...
    {
        using subtypes = std::tuple<property_instances_t...>;
    };
};
```

Usage:
```c++
using amp_mod_params = properties<props::input, props::freq, props::amp>;

auto params = amp_mod_params::make{ block_memory::make{buffer}, 440, cv<1.0> };

amplitude_modulation(params[props::input], params[props::freq], params[props::amp]);

static_assert(is_cv<params[prop::amp]>); // it's constant!
```

note:
15:11

---

### Detail #1: Property specs...

Outer struct holds **property specs** describing **underlying type, a unique type tag, and other options**.

```c++
namespace props
{
constexpr auto freq  = tag_spec{ type_code_v<double>, options{ tag_v<struct freq_tag> } };
constexpr auto amp   = tag_spec{ type_code_v<double>, options{ tag_v<struct amp_tag> } };
constexpr auto input = tag_spec{ type_code_v<block_memory<double>>, options{ tag_v<struct input_tag> } };
};
```

We will use these to access property values in our structure.

note:
18:36

---
### Detail 2: Accessing properties...

```c++
// in our outer class...
static constexpr auto specs = std::tuple<property_specs_t...>{}

// and in our inner `instance` class
using subtypes = std::tuple<property_instances_t>;

template <typename spec_t>
constexpr auto operator[](spec_t spec) const -> decltype(auto)
{
	constexpr auto index = matching_spec_index(spec, specs);
	using detected_t     = std::tuple_element_t<index, subtypes>;

	return static_cast<detected_t const&>(*this);
}

// remember:
//   template <typename... property_instances_t>
//   struct instance : property_instances_t... {
//       using subtypes = std::tuple<property_instances_t...>;
```

We just cast ourselves to the type of the property we're looking for and return.

---
### Detail #3:  Mixins....

Suppose we want `span.width()` instead of `span[props:width]`...

**We want to layer interface on top of our `properties::instance`.**

```c++
template <is_mixin_option auto mixin, typename... property_specs_t>
struct properties
{
    template <typename... property_instances_t>
    struct instance : 
		make_mixin<mixin>::type<instance<property_instances_t...>>,
	    property_instances_t...
    {
    };
};
```

---

```c++
template <typename properties_t>
struct size_accessor_mixin
{
    auto rows(this properties_t const& prop) -> decltype(auto)
    {
        return prop[props::rows];
    }

    auto cols(this properties_t const& prop) const -> decltype(auto)
    {
        return prop[props::cols];
    }
};
```

("Explicit object member functions (deducing this)")

---

### Some other cool details...

- ✓ individual `property` types can also get mixins...
- ✓ nested properties (e.g. `rect` type made up of `size` and `offset`, each `row` and `col`)
- ✓ mutating `properties` structures
- ✓ constant_value properties are `size==0` !

---

### Problem: Sometimes I want an array of `properties`, but sometimes I want a `properties` full of arrays...

```c++
using array_of_props = std::array<properties<...>, 100>
using prop_of_arrays = properties<tag_spec{type_code_v<std::array<float, 100>>, ...}
```

Think: do I want a N bundles of properties for my "voices", or an array of every `freq` property for every voice?

note:
25:14

---
### Solution: Property transforms...

```c++
constexpr auto new_size = size.transform(

	// First lambda: transform the spec (type)
    [](auto spec) constexpr
    {
	    // This is the original spec for the property...
	    // You can return the original or pass back a new one
	    auto new_type = type_code_v<std::array<spec.storage, 256>>;
	    return spec.with_storage(new_type);
    },
    
    // Second lambda: transform the value
    []<typename _t>(_t value, auto) constexpr
    {
        std::array<_t, 256> new_value;
        new_value.fill(value);
        return new_value;
    });
```

---
### What do we have now?

- We have a compile-time / runtime variant
- ... that can be passed around as a normal value
- ... and also used as a template argument (if cv)
- We have structured key-value pair storage with the same semantics
- Zero-cost "evaporating" storage for constants
- One `properties` *description* can be transformed into many kinds of concrete storage
#### What next?

---
# Part 1.5: Hoisting

note:
28:44

---
### **Problem**: we want to run a **specific version of our process function depending on a runtime condition**.  

We do this all the time already...
```c++
if (filter == filters::low_pass) {
	run_lowpass(input);
} else {
	run_hipass(input);
}
```

---

### Solution: Convert our runtime value to a cv<>

```c++
if (value == 0.0) { 
	some_function(cv<0.0>);
} else if (value == 1.0) {
	some_function(cv<1.0>);
} else {
	some_function(value);
}
```

Maybe even nested....
```c++
if (value_a == 0) {
   if (value_b == lowpass) {
      some_function(cv<0>, cv<lowpass>);
   }
}
```

---
### How do we implement (fast-forward version)...
### Part 1: Describe our hoist

```c++
// Turn 0.0 and 1.0 into cv<>, else just use the runtime value
constexpr auto amp_hoist  = hoist::values<props::amp, 0.0, 1.0>{};
```


---

### Part 2: Check the runtime value and forward

```c++ 
// After some setup stuff...
template <
	template <auto> typename func_t, 
	auto hoist_value, 
	auto... remaining_hoist_values, 
	class... args_t>
constexpr auto hoist_base_noswitch(auto value, args_t&&... args)
{
    if (value == hoist_value) 
    {   // we have a matching value!
        return func_t<hoist_value>{}(fwd(args)...);
    }
    else
    {
        if constexpr (sizeof...(remaining_hoist_values) > 0)
        {   // we more values to check....
            return hoist_base_noswitch<func_t, remaining_hoist_values...>(fwd(value), fwd(args)...);
        }
        else
        {  // we are out of values to check, so call with a no_hoist tag
            return func_t<no_hoist{}>{}(fwd(args)...);
        }
    }
}

```

(since we will do this for multiple properties, our `func_t` collects new property values and then forwards them to a callback....)

---
### Final implementation

```c++
using inputs_t =
    properties<props::input, props::freq, props::amp, props::waveform>;

auto amp_hoist      = hoist::values<props::amp, 0.0, 1.0>{};
auto waveform_hoist = hoist::values<props::waveform, sine, tri, saw, square>{};

// 
auto process_hoisted = hoist::hoisted(
    [](auto inputs) // <-- some of these are hopefully cv<> now!
    {
        return amplitude_modulation(
            inputs[props::input],
            inputs[props::freq],
            inputs[props::amp],
            inputs[props::waveform]);
    },
    amp_hoist,
    waveform_hoist);

auto inputs = inputs_t{ buffer, freq, amp, waveform };
auto result = process_hoisted(inputs);
```

---
### Other cool details...

- ✓ Integer hoists can use `switch`
- Since hoist details are just data, **you can empirically test what hoist values help performance** (at some point code size will slow you down even more...)

note:
33:40

---

### Problem: How do you divide up a runtime numeric range, into compile-time constant chunks of work?

```c++
for (auto i = 0; i < samples; i++) { ... }
```

---
### Solution: Range hoist
You've done this before:
```c++
auto chunks_of_four = voices / 4;
auto remainder = voices - four_chunks;
for (auto i = 0; i < four_chunks; i++) { process_four(...) }
for (auto i = 0; i < remainder; i++) { process_one(...) }
```

*(this is critical for SIMD code!)*

---

```c++
using props_t = properties<props::row_start, props::row_size, props::col_start, props::col_size>;

auto col_range = hoist::range{ props::col_start, props::col_size }; 

auto p = props_t::make(7, 11, 0, 0);

hoist::hoisted_range(
	[&](auto props)
	{
		// Called with power-of-two sub-partitions, stepping down to 1
		// col_start: 0, col_size: cv<4>
		// col_start: 4, col_size: cv<4>
		// col_start: 8, col_size: cv<2>
		// col_start: 10, col_size: cv<2>
		// col_start: 11, col_size: cv<1>
	},
	col_range)(p);

```

*I'm only doing power-of-two steps starting at a max, but there are many other ways to partition!*

---

### Other hoisting types

- **partitioned hoist** (execute a different graph based on numeric ranges - e.g. envelopes, rate < 1.0 vs rate > 1.0)


---
### What do we have now?
*It's not like writing if-statements is hard....*

- **We can hoist branches from inner-loop code to the beginning of our process call.**
- **Our inner-loop becomes more branch-less + inline-able** at the cost of overall code size.
- **Parametric**: we can experiment with hoisting strategies without changing our algorithm code.

note:
38:00

---

# Part 2: Graphs

![[graph-example.svg]]
note:
0:00

---
### How do we describe the operations we'll perform on audio?

Our goal is to produce meta-functions, e.g. **one description** of our inner-loop algorithm that we can compile different versions of, transform, and tweak.

---

**Directed graphs** are a great representation of how to compose the operations to process audio. 

![[_PROJECTS/GAUGE/graph-1.svg]]

We can operate on and transform graphs in predictable ways.

This will help us **build a bridge between expressive algorithm descriptions and concrete, optimized process functions**. 

---

### What should it look like?

```c++
template <typename value_t, typename... inputs_t>
struct node
{
	value_t value;
	std::tuple<inputs_t...> inputs;
};

// example:
constexpr auto input_1 = node{ read_op, buffer_1 };
constexpr auto input_2 = node{ read_op, buffer_2 };
constexpr auto mult = node{ multiply_op, input_1, input_2 };
```

---

### But what about this...

```c++
constexpr auto input_1 = node{ read_op, buffer_1 };
constexpr auto input_2 = node{ read_op, buffer_2 };
constexpr auto mult = node{ multiply_op, input_1, input_2 }; // <-- first use
constexpr auto add = node { add_op, mult, input_1 };         // <-- second use
```

*We're using `input_t` twice - but we can't take a reference to it at compile time, and if we copy we'll run the node twice...* 

---
### Solution: Track our nodes with unique ids according to their type

```c++
template <typename value_t, uid... input_ids_t>
struct node
{
    static constexpr uid id = unique_id<node>();

    const value_t value;
    static constexpr std::tuple input_ids{ input_ids_t... };
};
```

Our node's id is unique to it's type, which means **another node with the same value_t and inputs has the same id.**

---

### Problem: Where does our id come from?

`unique_id<node>()`

---

### Solution: Use a friend injection counter

```c++
// FRIEND INJECTION COUNTER
template <auto id>
struct counter
{
    using tag = counter;
    struct generator
    {
        friend consteval auto is_defined(tag)
        {
            return true;
        }
    };
    friend consteval auto is_defined(tag);

    template <typename tag_t = tag, auto = is_defined(tag_t{})>
    static consteval auto exists(auto)
    {
        return true;
    }
    static consteval auto exists(...)
    {
        return generator(), false;
    }
};

template <typename T, auto Id = int{0}>
consteval auto unique_id() -> u64
{
    if constexpr (counter<Id>::exists(Id))
    {
        return unique_id<T, Id + 1>();
    }
    else
    {
        return Id;
    }
}

using uid = decltype(unique_id<void>());

```

---

1. Each counter stores state using a **hidden friend function**, `is_defined`()

```c++
template <auto id>
struct counter
{
	using tag = counter;
	struct generator
	{
		friend consteval auto is_defined(tag) // <-- definition
		{
			return true;
		}
	};
// ...
// once we've defined counter<10>, the is_defined(counter<10>) -> true
```

---

2.  We test whether that friend `exists()` to detect prior use. We only hit the `true` version if `is_defined` is already defined, e.g. we've seen the id before.

```c++
// still inside of struct counter...
// SFINAE means we only hit this if our is_defined is already defined...
template <typename tag_t = tag, auto = is_defined(tag_t{})> 
static consteval auto exists(auto)
{
	return true;
}
static consteval auto exists(...)
{
    // if it didn't exist, we construct a generator() here, which defines it
	return generator(), false;
}
```

---

3.  For a given `unique_id<T>()`, we start at `id==0`, and increment until we find an id that is not yet defined (at compile time).

```c++
template <typename T, u64 Id = 0>
consteval auto unique_id() -> u64
{
    if constexpr (counter<Id>::exists(Id))
    {
        return unique_id<T, Id + 1>();
    }
    else
    {
        return Id;
    }
}
```

---
### Problem: We have an id, but how do we retrieve the type later?

Solution: another `friend` injection trick

```c++
template <auto key>
struct reader
{
    constexpr auto friend get(reader<key>);
};
	
template <auto key, typename value_t>
struct writer
{
    constexpr auto friend get(reader<key>)
    {
        return std::type_identity<value_t>{};
    }
};

template <uid node_id>
using node_from_id = typename decltype(get(reader<node_id>{}))::type;
```

1. Defining `writer<uid, type_t>` defines a friend `get` function
2. Calling that `get` function later with our `reader<key>` returns the type as an object

```c++
using original_node_t = node_from_id<node_id>;
auto node = original_node_t{};
```
---
### What we have now?

```c++
constexpr auto A = make_node(cv<1>);
constexpr auto B = make_node(add_op, A, cv<10>);
constexpr auto C = make_node(mul_op, A, cv<10>);
constexpr auto D = make_node(divide_op, B, C);

constexpr auto graph = make_graph(D);
```

**Limitation**: Our id-based system only works if our nodes are empty, e.g. if we can reconstruct them from an id by doing `original_node_t{}`. 

But, our constant-value system already makes passing data to and from template space easy, so this isn't a problem.

---

### What can we do with our graph?

Let's envision a few core transformations and see how far we get with them...

---
### Operation #1:  map


![[graph-map.svg]]
Visit each node in a graph (or graphs). Pass the nodes value to function, 
and construct a new graph from the results. (could also be called zip...)

```c++
auto new_graph = digraph::visit_map([](auto value_1, auto value_2) {
	return std::tuple{ value_1, value_2 };
}, graph_1, graph_2)
```

---
### Operation #2: fold

![[graph-fold.svg]]

Depth-first traversal, visit each node with the result 

```c++
auto result = digraph::visit_fold([](auto value, auto... inputs) {
	return value(inputs...); // assuming each node value is a function
}, graph)
```

---

### Variations

- `visit_fold_all` (same but return a tuple of all node outputs)
- `visit_fold_transform` (same, but return a new graph with the outputs of each fold step)
- `visit_fold_reverse` (same, but traverse in reverse, e.g. visit each node with its output nodes)
- `visit_fold_stateful` (same but nodes output both a value and a state - return the state as a new graph, which can be passed into future `visit_fold_stateful` calls)
- `visit_fold_graph` (same but pass the actual nodes, not their values, and turn the result into a new graph - this can change the graph shape)

---

# What can we do?


---
### Inject external inputs

```c++
template <auto input_spec>
struct input_node {
	static constexpr auto spec = input_spec;
};

auto sig = nodes::sin_osc(input_node<props::freq>{});

auto inputs = input_properties::make(440); // <-- runtime value, e.g. from DAW

auto new_graph = visit_map([&](auto value) {
   if constexpr (is_input_node<decltype(value)>) {
	   return inputs[value.spec];
   } else {
	   return value;
   }
});
```

---
### Stateful nodes

```c++
struct step_node {
	struct state { int i = 0; };
	auto initial_state() { return state{}; }
	auto operator(auto state, auto... inputs) {
		auto value = state.i;
		state.i++;
	    return std::pair(value, state);
	}
};

// same shape as our graph...
auto initial_state = visit_fold_transform([](auto node) {
	return node.initial_state();
}, graph);

// iterate the state once
auto [result, next_state] = visit_fold_stateful([](){...}, graph, initial_state);
```

---
### Other interesting operations

- **Inter-graph feedback** (this can be done the same way as inputs)
- **Hoisting** (create different transformed graphs based on a runtime value)
- 

note:
17:00

---

# Part 3: Vectorization

![[vectorization.svg]]

---

![[vectorization.svg]]
SIMD vectorization of audio processing usually happens along one of two axes: **time**, or **voices**.

- Some operations are faster along one axis or another. 
- Some operations CAN'T be vectorized along one axis or another.

---

**For example:**
- **time vectorization** works well for memory-bound operations like delays, buffer playback (reading sequential memory is fastest)
- **voice vectorization** is required when the value at the current time depends on very recent values (e.g. IIR filters, short delays)

**The execution plan interacts with cache,  CPU registers, branch prediction, etc. in complex ways.**

---

### Problem: For a complex graphs, the best execution plan / vectorization is non-obvious and can't be determined a priori

Moreover: we don't want to have to change our graph description just to test our a theory about e.g. cache misses or brach misprediction.

---

### Solution: Our graph describes operations on M x N (voices x time) vectors

We describe our execution plan for those vectors **separately** from the graph.

---
### Basic implementation

```c++
// MSVC doesn't support ofc, but we can imagine alternatives...
template <typename T, size_t size>
using vector = T __attribute__((__vector_size__(sizeof(T) * size)));

template <typename T, size_t rows_v, size_t cols_v, axis storage_axis_v>
struct block
{
    vector<T, rows_v * cols_v> value;
};

template <typename T>
using block_memory = properties<
		detail::memory_mixin<T>::block_option, // <-- memory access functions mixin
		props::data_ptr<T>, 
		props::alignment, 
		props::rows, props::cols, 
		props::row_stride, props::col_stride>;
```

---

### Operating on blocks

```c++
template <simd::is_block type>
auto operator/(type lhs, type rhs)
{
    return type{ lhs.value / rhs.value }; // clang helps us with basic math ops...
}

auto divided = numer_block / denom_block;
```

---

### Operating on external memory

```c++
template <typename derived_t>
struct block_memory_mixin
{
    template <size_t result_rows, size_t result_cols, axis result_axis = axis::col>
    constexpr auto load_block(auto row_offset, auto col_offset) const 
	    -> simd::block<T, result_rows, result_cols, result_axis>;
};

auto 4_samples_1_voice = memory.load_block<1, 4>(0, 0);
auto 4_voices_1_sample = memory.load_block<4, 1>(0, 0);
```

---

### Problem: SIMD loads have to be aligned (sort of...)

I have to load SIMD values from aligned addresses. On Intel esp getting this wrong is especially bad ☠️☠️☠️

But there's no compile-time concept of an "aligned pointer".

---

### Solution: constant_value to the rescue...

1. When we receive memory from the *outside* (e.g. a buffer from our host application), we validate it's aligned, and **then set a `cv<>` alignment property on our `block_memory`**:
   ```
   auto alignment = cv<0>; // aligned
   auto mem = block_memory<float>::make(buffer, alignment, ...);
   ```

Our alignment starts as a `constant_value`. **As LONG as our increments / memory offsets are also `constant_value`'s, alignment continues to be cv<>.**

Since we almost always process a graph in fixed sizes (e.g. a power-of-two block size), our alignment will stay cv<> - so we get zero-cost alignment correctness built into our type system.

---

### Problem: What if e.g. our processing block size is a runtime value (which often the case...)?

---

### Solution:  range hoist

Range hoist to break processing of `numSamples` up into `constant_value` block sizes.

---

### Problem: My graph is complex, I have no idea what execution plan is more efficient

For example: I'm reading from 8 buffers, and processing them each with IIR filters. Buffers reads should be time-parallel, but IIR filters must be voice-parallel.

---

### Solution: Parameterize and test

*(interpolation_benchmarks.cpp)*


---

# Subgraphs

We can process parts of our graph with a different 

```c++
// Process `count` voices, 4 voices at a time, 1 sample at a time
// ... and sum the result
// voice_graph() describes the graph of each voice
constexpr auto sum = n::subgraph_sum<4, 1>(count, n::cursor(), voice_graph())();
```

---

### What do we have now?

- A way of describing operations on audio that is independent of how we process it (by voice, by time)
- A building block (`simd::block`) to build a library of fast vectorized operations
- The ability to experiment with different execution plans to test performance, without changing code.

---
### Going further...

When we describe our audio graph this way, **we begin to collapse buffer-based processing, wavetable synthesis, additive synthesis, granular synthesis, wave terrain, etc. into almost identical operations**.

---

### Whats next?

- The goal for the prototype library is to **produce basic additive synthesis and granular synthesis generators. These are simple to write, and we can test optimization and validity.
- These example generators should also give guidance on syntax and feature set.
- Once these are finished, I'll probably re-write as 2-3 related libraries (e.g. one for constant values), taking the most useful patterns from the prototype and ditching the rest.
