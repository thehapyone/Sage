
# Example usage
yaml_config = """
name: GameStartup

process: hierarchical

agents:
  - role: Game Designer
    goal: Design engaging and innovative game mechanics
    backstory: An expert with over a decade of experience in game design and is known for creating unique and popular game mechanics.

  - role: Marketing Strategist
    goal: Develop a marketing strategy to launch the game successfully
    backstory: You have worked on several successful game launches and excels at creating buzz and engaging the gaming community.

tasks:
  - description: "You help research and design the core mechanics of games. This is the game instructions: {input}"
    agent: Game Designer
    expected_output: A detailed report on the game mechanics including sketches and flowcharts

  - description: "Conduct a competitor analysis for similar games. Game details: {input}"
    agent: Marketing Strategist
    expected_output: A report on competitor strengths, weaknesses, and market positioning

  - description: "You develop the initial concept art and prototypes for the game. Game details: {input}"
    agent: Game Designer
    expected_output: Concept art and prototype sketches

  - description: "You wil create a comprehensive marketing plan for the game launch. Game details: {input}"
    agent: Marketing Strategist
    expected_output: A complete marketing strategy document with timelines, channels, and key messages
"""
