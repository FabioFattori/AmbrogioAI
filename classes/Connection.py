class Connection:
    def __init__(self, to , weight, from_):
        self.to = to
        self.weight = weight
        self.from_ = from_
        
    def __str__(self) -> str:
        return f"Connection from {self.from_.id} to {self.to.id} with weight {self.weight}"
    
    def __eq__(self, value: object) -> bool:
        return self.to == value.to and self.from_ == value.from_