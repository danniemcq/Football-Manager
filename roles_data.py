# Football Manager roles and their attribute requirements
roles_data = {
    # Keepers
    'Goalkeeper': {
        'Goalkeeping': {
            'Aerial Reach': 3, 'Command Of Area': 3, 'Communication': 3,
            'Eccentricity': 1, 'First Touch': 1, 'Handling': 3,
            'Kicking': 3, 'One On Ones': 3, 'Passing': 1,
            'Punching (Tendency)': 1, 'Reflexes': 5, 'Rushing Out (Tendency)': 3,
            'Throwing': 3
        },
        'Mental': {
            'Aggression': 3, 'Anticipation': 1, 'Bravery': 3,
            'Composure': 1, 'Concentration': 3, 'Decision Making': 3,
            'Determination': 5, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 1, 'Positioning': 3, 'Teamwork': 5,
            'Vision': 1, 'Work Rate': 1
        },
        'Physical': {
            'Acceleration': 1, 'Agility': 3, 'Balance': 3,
            'Jumping Reach': 5, 'Natural Fitness': 3, 'Pace': 1,
            'Stamina': 3, 'Strength': 5
        }
    },
    'Sweeper Keeper': {
        'Goalkeeping': {
            'Aerial Reach': 3, 'Command Of Area': 5, 'Communication': 5,
            'Eccentricity': 3, 'First Touch': 3, 'Handling': 3,
            'Kicking': 3, 'One On Ones': 5, 'Passing': 3,
            'Punching (Tendency)': 1, 'Reflexes': 5, 'Rushing Out (Tendency)': 5,
            'Throwing': 3
        },
        'Mental': {
            'Aggression': 3, 'Anticipation': 5, 'Bravery': 5,
            'Composure': 3, 'Concentration': 3, 'Decision Making': 5,
            'Determination': 3, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 1, 'Positioning': 5, 'Teamwork': 3,
            'Vision': 3, 'Work Rate': 1
        },
        'Physical': {
            'Acceleration': 3, 'Agility': 3, 'Balance': 3,
            'Jumping Reach': 3, 'Natural Fitness': 3, 'Pace': 3,
            'Stamina': 1, 'Strength': 3
        }
    },
    # Defenders
    'Ball Playing Defender (Defend)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 1, 'Finishing': 1,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 5,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 5,
            'Passing': 3, 'Penalty Taking': 1, 'Tackling': 5,
            'Technique': 3
        },
        'Mental': {
            'Aggression': 3, 'Anticipation': 5, 'Bravery': 5,
            'Composure': 3, 'Concentration': 5, 'Decision Making': 5,
            'Determination': 3, 'Flair': 1, 'Leadership': 3,
            'Off The Ball': 1, 'Positioning': 5, 'Teamwork': 3,
            'Vision': 3, 'Work Rate': 3
        },
        'Physical': {
            'Acceleration': 3, 'Agility': 1, 'Balance': 1,
            'Jumping Reach': 5, 'Natural Fitness': 3, 'Pace': 3,
            'Stamina': 3, 'Strength': 5
        }
    },
    'Central Defender (Defend)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 1, 'Finishing': 1,
            'First Touch': 1, 'Free Kick Taking': 1, 'Heading': 5,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 5,
            'Passing': 1, 'Penalty Taking': 1, 'Tackling': 5,
            'Technique': 1
        },
        'Mental': {
            'Aggression': 3, 'Anticipation': 5, 'Bravery': 5,
            'Composure': 3, 'Concentration': 5, 'Decision Making': 3,
            'Determination': 3, 'Flair': 1, 'Leadership': 3,
            'Off The Ball': 1, 'Positioning': 5, 'Teamwork': 3,
            'Vision': 1, 'Work Rate': 3
        },
        'Physical': {
            'Acceleration': 1, 'Agility': 1, 'Balance': 1,
            'Jumping Reach': 5, 'Natural Fitness': 3, 'Pace': 3,
            'Stamina': 3, 'Strength': 5
        }
    },
    'Full-Back (Defend)': {
        'Technical': {
            'Corners': 1, 'Crossing': 3, 'Dribbling': 1, 'Finishing': 1,
            'First Touch': 1, 'Free Kick Taking': 1, 'Heading': 3,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 5,
            'Passing': 3, 'Penalty Taking': 1, 'Tackling': 5,
            'Technique': 1
        },
        'Mental': {
            'Aggression': 3, 'Anticipation': 3, 'Bravery': 3,
            'Composure': 1, 'Concentration': 3, 'Decision Making': 3,
            'Determination': 3, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 1, 'Positioning': 5, 'Teamwork': 3,
            'Vision': 1, 'Work Rate': 3
        },
        'Physical': {
            'Acceleration': 3, 'Agility': 1, 'Balance': 1,
            'Jumping Reach': 3, 'Natural Fitness': 3, 'Pace': 3,
            'Stamina': 3, 'Strength': 3
        }
    },
    'Wing-Back (Defend)': {
        'Technical': {
            'Corners': 1, 'Crossing': 3, 'Dribbling': 3, 'Finishing': 1,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 1,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 3,
            'Passing': 3, 'Penalty Taking': 1, 'Tackling': 3,
            'Technique': 3
        },
        'Mental': {
            'Aggression': 1, 'Anticipation': 3, 'Bravery': 3,
            'Composure': 1, 'Concentration': 3, 'Decision Making': 3,
            'Determination': 3, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 3, 'Positioning': 3, 'Teamwork': 3,
            'Vision': 1, 'Work Rate': 5
        },
        'Physical': {
            'Acceleration': 3, 'Agility': 3, 'Balance': 3,
            'Jumping Reach': 1, 'Natural Fitness': 3, 'Pace': 5,
            'Stamina': 5, 'Strength': 1
        }
    },
    # Midfielders
    'Central Midfielder (Defend)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 1, 'Finishing': 1,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 1,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 3,
            'Passing': 3, 'Penalty Taking': 1, 'Tackling': 3,
            'Technique': 1
        },
        'Mental': {
            'Aggression': 3, 'Anticipation': 3, 'Bravery': 3,
            'Composure': 3, 'Concentration': 3, 'Decision Making': 3,
            'Determination': 3, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 1, 'Positioning': 5, 'Teamwork': 3,
            'Vision': 1, 'Work Rate': 3
        },
        'Physical': {
            'Acceleration': 1, 'Agility': 1, 'Balance': 1,
            'Jumping Reach': 1, 'Natural Fitness': 3, 'Pace': 1,
            'Stamina': 3, 'Strength': 3
        }
    },
    'Box To Box Midfielder (Support)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 3, 'Finishing': 3,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 1,
            'Long Shots': 3, 'Long Throws': 1, 'Marking': 1,
            'Passing': 3, 'Penalty Taking': 1, 'Tackling': 3,
            'Technique': 3
        },
        'Mental': {
            'Aggression': 3, 'Anticipation': 3, 'Bravery': 3,
            'Composure': 3, 'Concentration': 3, 'Decision Making': 3,
            'Determination': 3, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 3, 'Positioning': 3, 'Teamwork': 3,
            'Vision': 1, 'Work Rate': 5
        },
        'Physical': {
            'Acceleration': 3, 'Agility': 1, 'Balance': 1,
            'Jumping Reach': 1, 'Natural Fitness': 3, 'Pace': 3,
            'Stamina': 5, 'Strength': 3
        }
    },
    'Defensive Midfielder (Defend)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 1, 'Finishing': 1,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 3,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 5,
            'Passing': 3, 'Penalty Taking': 1, 'Tackling': 5,
            'Technique': 1
        },
        'Mental': {
            'Aggression': 3, 'Anticipation': 5, 'Bravery': 3,
            'Composure': 3, 'Concentration': 5, 'Decision Making': 3,
            'Determination': 3, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 1, 'Positioning': 5, 'Teamwork': 3,
            'Vision': 1, 'Work Rate': 3
        },
        'Physical': {
            'Acceleration': 1, 'Agility': 1, 'Balance': 1,
            'Jumping Reach': 3, 'Natural Fitness': 3, 'Pace': 1,
            'Stamina': 3, 'Strength': 3
        }
    },
    'Deep Lying Playmaker (Defend)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 1, 'Finishing': 1,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 1,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 1,
            'Passing': 5, 'Penalty Taking': 1, 'Tackling': 3,
            'Technique': 3
        },
        'Mental': {
            'Aggression': 1, 'Anticipation': 3, 'Bravery': 1,
            'Composure': 5, 'Concentration': 3, 'Decision Making': 5,
            'Determination': 3, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 1, 'Positioning': 3, 'Teamwork': 3,
            'Vision': 5, 'Work Rate': 1
        },
        'Physical': {
            'Acceleration': 1, 'Agility': 1, 'Balance': 1,
            'Jumping Reach': 1, 'Natural Fitness': 3, 'Pace': 1,
            'Stamina': 3, 'Strength': 1
        }
    },
    'Mezzala (Support)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 3, 'Finishing': 1,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 1,
            'Long Shots': 3, 'Long Throws': 1, 'Marking': 1,
            'Passing': 3, 'Penalty Taking': 1, 'Tackling': 1,
            'Technique': 3
        },
        'Mental': {
            'Aggression': 1, 'Anticipation': 3, 'Bravery': 3,
            'Composure': 3, 'Concentration': 1, 'Decision Making': 3,
            'Determination': 3, 'Flair': 3, 'Leadership': 1,
            'Off The Ball': 3, 'Positioning': 1, 'Teamwork': 3,
            'Vision': 3, 'Work Rate': 3
        },
        'Physical': {
            'Acceleration': 3, 'Agility': 3, 'Balance': 3,
            'Jumping Reach': 1, 'Natural Fitness': 3, 'Pace': 3,
            'Stamina': 3, 'Strength': 1
        }
    },
    # Forwards
    'Advanced Forward (Attack)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 3, 'Finishing': 5,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 3,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 1,
            'Passing': 1, 'Penalty Taking': 3, 'Tackling': 1,
            'Technique': 3
        },
        'Mental': {
            'Aggression': 3, 'Anticipation': 3, 'Bravery': 3,
            'Composure': 3, 'Concentration': 3, 'Decision Making': 3,
            'Determination': 3, 'Flair': 3, 'Leadership': 1,
            'Off The Ball': 5, 'Positioning': 1, 'Teamwork': 1,
            'Vision': 1, 'Work Rate': 3
        },
        'Physical': {
            'Acceleration': 3, 'Agility': 3, 'Balance': 3,
            'Jumping Reach': 3, 'Natural Fitness': 3, 'Pace': 3,
            'Stamina': 3, 'Strength': 3
        }
    },
    'Complete Forward (Support)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 3, 'Finishing': 3,
            'First Touch': 5, 'Free Kick Taking': 1, 'Heading': 3,
            'Long Shots': 3, 'Long Throws': 1, 'Marking': 1,
            'Passing': 3, 'Penalty Taking': 1, 'Tackling': 1,
            'Technique': 5
        },
        'Mental': {
            'Aggression': 3, 'Anticipation': 3, 'Bravery': 3,
            'Composure': 3, 'Concentration': 3, 'Decision Making': 3,
            'Determination': 3, 'Flair': 3, 'Leadership': 1,
            'Off The Ball': 3, 'Positioning': 1, 'Teamwork': 3,
            'Vision': 3, 'Work Rate': 3
        },
        'Physical': {
            'Acceleration': 3, 'Agility': 3, 'Balance': 3,
            'Jumping Reach': 3, 'Natural Fitness': 3, 'Pace': 3,
            'Stamina': 3, 'Strength': 5
        }
    },
    'Deep Lying Forward (Support)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 3, 'Finishing': 3,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 1,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 1,
            'Passing': 3, 'Penalty Taking': 1, 'Tackling': 1,
            'Technique': 3
        },
        'Mental': {
            'Aggression': 1, 'Anticipation': 3, 'Bravery': 1,
            'Composure': 3, 'Concentration': 3, 'Decision Making': 3,
            'Determination': 3, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 3, 'Positioning': 1, 'Teamwork': 3,
            'Vision': 3, 'Work Rate': 1
        },
        'Physical': {
            'Acceleration': 1, 'Agility': 1, 'Balance': 1,
            'Jumping Reach': 1, 'Natural Fitness': 3, 'Pace': 1,
            'Stamina': 3, 'Strength': 3
        }
    },
    'Poacher (Attack)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 1, 'Finishing': 5,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 3,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 1,
            'Passing': 1, 'Penalty Taking': 3, 'Tackling': 1,
            'Technique': 3
        },
        'Mental': {
            'Aggression': 1, 'Anticipation': 5, 'Bravery': 1,
            'Composure': 3, 'Concentration': 3, 'Decision Making': 3,
            'Determination': 3, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 5, 'Positioning': 1, 'Teamwork': 1,
            'Vision': 1, 'Work Rate': 1
        },
        'Physical': {
            'Acceleration': 3, 'Agility': 1, 'Balance': 1,
            'Jumping Reach': 3, 'Natural Fitness': 3, 'Pace': 3,
            'Stamina': 1, 'Strength': 1
        }
    },
    'Pressing Forward (Defend)': {
        'Technical': {
            'Corners': 1, 'Crossing': 1, 'Dribbling': 1, 'Finishing': 3,
            'First Touch': 3, 'Free Kick Taking': 1, 'Heading': 3,
            'Long Shots': 1, 'Long Throws': 1, 'Marking': 1,
            'Passing': 1, 'Penalty Taking': 1, 'Tackling': 3,
            'Technique': 1
        },
        'Mental': {
            'Aggression': 5, 'Anticipation': 3, 'Bravery': 3,
            'Composure': 1, 'Concentration': 3, 'Decision Making': 3,
            'Determination': 3, 'Flair': 1, 'Leadership': 1,
            'Off The Ball': 3, 'Positioning': 1, 'Teamwork': 3,
            'Vision': 1, 'Work Rate': 5
        },
        'Physical': {
            'Acceleration': 3, 'Agility': 1, 'Balance': 3,
            'Jumping Reach': 1, 'Natural Fitness': 3, 'Pace': 3,
            'Stamina': 5, 'Strength': 3
        }
    }
}