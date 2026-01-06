library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;



entity MAC is
    Port (clk : in std_logic;
          reset : in std_logic;
          en : in std_logic;
          a : in signed(7 downto 0);
          b : in signed(7 downto 0);
          acc_reg : out signed(15 downto 0));
end MAC;

architecture Behavioral of MAC is
signal acc_reg1 : signed(15 downto 0) := (others => '0');
begin

    process(clk)
    begin
    
    if (rising_edge(clk)) then
        if reset = '1' then
            acc_reg1 <= (others => '0');
        elsif en = '1' then 
            acc_reg1 <= acc_reg1 + resize(a * b,16);
        end if;
     end if;
     
    end process;

    acc_reg <= acc_reg1;

end Behavioral;
