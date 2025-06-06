# Contributing to Transformer for Metaheuristic Algorithm Selection

We welcome contributions to this research project! This guide will help you get started.

## 🎯 **How to Contribute**

### **Types of Contributions**
- 🐛 **Bug fixes** - Fix issues in existing code
- ✨ **New features** - Add new algorithms, benchmarks, or models
- 📚 **Documentation** - Improve README, add examples, write tutorials
- 🧪 **Testing** - Add unit tests, improve test coverage
- 📊 **Research** - Add new datasets, analysis, or experimental results
- 🚀 **Performance** - Optimize algorithms or data collection

---

## 🛠️ **Development Setup**

### **1. Fork & Clone**
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Transformer-for-Metaheuristic-Algorithm-Selection.git
cd Transformer-for-Metaheuristic-Algorithm-Selection
```

### **2. Set Up Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy jupyter
```

### **3. Verify Setup**
```bash
# Run tests to ensure everything works
python test_basic_functionality.py
python test_new_benchmarks.py
```

---

## 📝 **Contribution Guidelines**

### **Code Style**
- Use **Black** for code formatting: `black .`
- Follow **PEP 8** style guidelines
- Use type hints where possible
- Write clear, descriptive variable names
- Add docstrings to functions and classes

### **Testing**
- Add tests for new functionality
- Ensure all existing tests pass
- Aim for >80% test coverage
- Use descriptive test names

### **Documentation**
- Update README if adding new features
- Add docstrings with examples
- Comment complex algorithms
- Update requirements.txt if adding dependencies

---

## 🔄 **Contribution Process**

### **1. Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

### **2. Make Changes**
- Implement your changes
- Add tests
- Update documentation
- Ensure code style compliance

### **3. Test Changes**
```bash
# Run all tests
python -m pytest tests/

# Run specific tests
python test_basic_functionality.py
python test_new_benchmarks.py

# Check code style
black --check .
flake8 .
```

### **4. Commit Changes**
```bash
git add .
git commit -m "Add: Brief description of your changes

- Detailed explanation if needed
- List key changes
- Reference issues if applicable"
```

### **5. Push & Create PR**
```bash
git push origin feature/your-feature-name
```
Then create a Pull Request on GitHub.

---

## 🎯 **Specific Contribution Areas**

### **🧮 Adding New Algorithms**
1. Create new file in `src/metaheuristics/`
2. Inherit from base optimizer class
3. Implement required methods
4. Add comprehensive tests
5. Update documentation

Example structure:
```python
class YourAlgorithm(BaseOptimizer):
    def __init__(self, ...):
        # Initialize parameters
    
    def optimize(self, objective_function, dimension, bounds, ...):
        # Main optimization logic
        return OptimizationResult(...)
```

### **📏 Adding New Benchmark Functions**
1. Add to `src/benchmarks/continuous_functions.py`
2. Inherit from `OptimizationFunction`
3. Implement `_evaluate()` method
4. Add to function registry
5. Include tests and validation

### **🤖 Improving Models**
1. Enhance transformer architecture
2. Add new baseline models
3. Improve feature extraction
4. Optimize hyperparameters

### **📊 Data & Analysis**
1. Add new problem instances
2. Improve data collection efficiency
3. Create analysis notebooks
4. Generate visualizations

---

## 🏷️ **Issue Labels**

When creating issues, use appropriate labels:
- `bug` - Something isn't working
- `enhancement` - New feature or improvement
- `documentation` - Documentation needs improvement
- `good first issue` - Good for newcomers
- `help wanted` - Need community help
- `research` - Research-related contributions

---

## 🧪 **Testing Guidelines**

### **Unit Tests**
```python
def test_algorithm_basic_functionality():
    """Test that algorithm runs without errors."""
    algorithm = YourAlgorithm()
    result = algorithm.optimize(sphere_function, 10, (-5, 5))
    assert result.success
    assert len(result.best_solution) == 10
```

### **Integration Tests**
```python
def test_data_collection_pipeline():
    """Test complete data collection workflow."""
    collector = DataCollector(target_runs=2)
    dataset = collector.run_collection()
    assert len(dataset) > 0
    assert 'best_fitness' in dataset.columns
```

---

## 📊 **Research Contributions**

### **Dataset Contributions**
- Follow existing data format
- Include comprehensive metadata
- Ensure reproducibility
- Document collection methodology

### **Analysis Contributions**
- Use Jupyter notebooks for exploration
- Create clear visualizations
- Include statistical tests
- Document findings clearly

### **Model Contributions**
- Benchmark against existing baselines
- Include ablation studies
- Document hyperparameter choices
- Provide training scripts

---

## 🔍 **Code Review Process**

### **Review Criteria**
1. **Functionality** - Does it work as intended?
2. **Code Quality** - Is it well-written and maintainable?
3. **Testing** - Are there adequate tests?
4. **Documentation** - Is it properly documented?
5. **Performance** - Does it meet performance requirements?

### **Reviewer Guidelines**
- Be constructive and respectful
- Explain reasoning for requested changes
- Test the changes locally if possible
- Check for edge cases and error handling

---

## 🤝 **Community Guidelines**

### **Be Respectful**
- Use inclusive language
- Be patient with newcomers
- Provide constructive feedback
- Acknowledge others' contributions

### **Communication**
- Use GitHub Issues for bugs and features
- Use GitHub Discussions for questions
- Be clear and specific in communications
- Search existing issues before creating new ones

---

## 🏆 **Recognition**

Contributors will be:
- Listed in repository contributors
- Mentioned in release notes for significant contributions
- Acknowledged in research publications (for research contributions)
- Added to AUTHORS file for major contributions

---

## 📞 **Getting Help**

- **Questions**: Use GitHub Discussions
- **Bugs**: Create GitHub Issues
- **Features**: Propose in GitHub Issues
- **Chat**: Contact maintainers directly

---

## 📄 **License**

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to advancing metaheuristic algorithm selection research! 🚀** 