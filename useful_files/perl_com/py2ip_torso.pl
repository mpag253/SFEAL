#!/usr/bin/perl

=pod

This script converts PCA-output mesh into an IPNODE format file for CMISS. 

Input = mesh from PCA result
Output = mesh for CMISS (.ipnode)


Created on Thu May 14 17:23 2015
@author: m.osanlouy@auckland.ac.nz


=cut

use strict;
use POSIX;


my $nfile = $ARGV[0]; 
my $ofile = $ARGV[1]; 


#######
### Process PCA-output mesh:
#######

# ****************** TORSO VERSION UNTESTED/INCOMPLETE *****************************************

open IPNODE, "<$nfile" or die "\033[31mError: Can't open $nfile\033[0m ";
my ($node, $i, @xyz, $nv, @deriv, $counter, @arr);

my $line = <IPNODE>;
$counter = 0;
while ($line) {
    if ($counter % 3 == 0) {
        
        if ($line =~ /(\d+.1)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
      
            $node = floor($1);
            $nv = 1;
            $i = 1;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
            #~ print "$1 $2 $3 \n";
            
            #~ print "$node $xyz[$node][$i] \n";
            
        } elsif ($line =~ /(\d+.2)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 2;
            $i = 1;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            #print "$1 $2 $3 \n";
            #~ print "$node $xyz[$node][0] $deriv[$node][$nv][$i][0] \n";
            
        } elsif ($line =~ /(\d+.3)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 3;
            $i = 1;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } elsif ($line =~ /(\d+.4)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 4;
            $i = 1;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } elsif ($line =~ /(\d+.5)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 5;
            $i = 1;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } elsif ($line =~ /(\d+.6)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 6;
            $i = 1;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } else {
            
            @arr = split(/ /, $line);
            $node = $arr[0];
            #~ print "$node \n";
            $nv = 0;
            $i = 1;
            $xyz[$node][$i] = $arr[1];
            $deriv[$node][$nv][$i][0] = $arr[2];
            $deriv[$node][$nv][$i][1] = $arr[3];
            $deriv[$node][$nv][$i][2] = $arr[4];     
            #~ print "$node $xyz[$node][$i] $deriv[$node][$nv][$i][0] $deriv[$node][$nv][$i][1] $deriv[$node][$nv][$i][2] \n";

        }
    } elsif ($counter % 3 == 1) {
        
        if ($line =~ /(\d+.1)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
      
            $node = floor($1);
            $nv = 1;
            $i = 2;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
            #~ print "$node $i $xyz[$node][0] $deriv[$node][$nv][$i][0] \n";
            
        } elsif ($line =~ /(\d+.2)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 2;
            $i = 2;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } elsif ($line =~ /(\d+.3)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 3;
            $i = 2;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } elsif ($line =~ /(\d+.4)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 4;
            $i = 2;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } elsif ($line =~ /(\d+.5)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 5;
            $i = 2;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } elsif ($line =~ /(\d+.6)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 6;
            $i = 2;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } else {
            
            @arr = split(/ /, $line);
            $node = $arr[0];
            #~ print "$node \n";
            $nv = 0;
            $i = 2;
            $xyz[$node][$i] = $arr[1];
            $deriv[$node][$nv][$i][0] = $arr[2];
            $deriv[$node][$nv][$i][1] = $arr[3];
            $deriv[$node][$nv][$i][2] = $arr[4];
            #~ print "$node $xyz[$node][$i] $deriv[$node][$nv][$i][0] $deriv[$node][$nv][$i][1] $deriv[$node][$nv][$i][2] \n";

        }
            
    } elsif ($counter % 3 == 2) {
        
        if ($line =~ /(\d+.1)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            #~ print "$line \n";
      
            $node = floor($1);
            $nv = 1;
            $i = 3;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
            #~ print "$1 $2 \n";
            
        } elsif ($line =~ /(\d+.2)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 2;
            $i = 3;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } elsif ($line =~ /(\d+.3)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 3;
            $i = 3;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } elsif ($line =~ /(\d+.4)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 4;
            $i = 3;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } elsif ($line =~ /(\d+.5)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 5;
            $i = 3;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
            #~ print "$1 $2 \n";
            
        } elsif ($line =~ /(\d+.6)\s(.*)\s(.*)\s(.*)\s(.*)\s(.*)/) {
            
            $node = floor($1);
            $nv = 6;
            $i = 3;
            $xyz[$node][$i] = $2;
            $deriv[$node][$nv][$i][0] = $3;
            $deriv[$node][$nv][$i][1] = $4;
            $deriv[$node][$nv][$i][2] = $5;
            
        } else {
            
            @arr = split(/ /, $line);
            $node = $arr[0];
            #~ print "$node \n";
            $nv = 0;
            $i = 3;
            $xyz[$node][$i] = $arr[1];
            $deriv[$node][$nv][$i][0] = $arr[2];
            $deriv[$node][$nv][$i][1] = $arr[3];
            $deriv[$node][$nv][$i][2] = $arr[4];  
            #~ print "$node $xyz[$node][$i] $deriv[$node][$nv][$i][0] $deriv[$node][$nv][$i][1] $deriv[$node][$nv][$i][2] \n";
        }
        
    }
        

    $line = <IPNODE>;
    $counter ++;   
}

close IPNODE;

open IPNODE, ">$ofile" or die "\033[31mError: Can't open $ofile\033[0m ";

print IPNODE " CMISS Version 1.21 ipnode File Version 2\n";
print IPNODE " Heading: Torso_pca_reconstructed\n\n";
printf IPNODE " The number of nodes is [ 112]:  112 \n\n";
print IPNODE " Number of coordinates [ 3]:  3\n";
print IPNODE " Do you want prompting for different versions of nj=1 [N]? Y\n";
print IPNODE " Do you want prompting for different versions of nj=2 [N]? Y\n";
print IPNODE " Do you want prompting for different versions of nj=3 [N]? Y\n";
print IPNODE " The number of derivatives for coordinate 1 is [0]: 3 \n";
print IPNODE " The number of derivatives for coordinate 2 is [0]: 3 \n";
print IPNODE " The number of derivatives for coordinate 3 is [0]: 3 \n";


#######
### Node info:
#######

my (@versions,@Nodes,@base);

@Nodes = (97..208);

for my $i (97..208){
    $versions[$i] = 0;
}

# Nodes with 1 version
for my $i (97..208){
    $versions[$i] = 1;
}

##Nodes with 2 versions
#for my $i (){
#    $versions[$i] = 2;
#}
#
# Nodes with 3 versions
#for my $i (){
#	$versions[$i] = 3;
#}
#
## Nodes with 4 versions
#for my $i (){
#	$versions[$i] = 4 
#}
#
## Nodes with 5 versions
#for my $i (){
#	$versions[$i] = 5; 
#}
#
## Nodes with 6 versions
#for my $i (){
#	$versions[$i] = 6; 
#}

&PrintModifiedNodes;

###########
### PRINT OUT THE MODIFIED NODES
###########
sub PrintModifiedNodes {

    foreach my $i (@Nodes){
        printf IPNODE "\n Node number [%5d]: %5d\n",$i,$i;
        if($versions[$i] == 0){
            $nv = 0;
            for my $j (1..3){
                print IPNODE " The number of versions for nj=$j is [1]: 1\n";
                print IPNODE " The Xj($j) coordinate is [ 0.00000E+00] CORRECT:   $xyz[$i][$j]\n"; 
                print IPNODE " The derivative wrt direction 1 is [ 0.00000E+00] CORRECT: $deriv[$i][$nv][$j][0]\n";
                print IPNODE " The derivative wrt direction 2 is [ 0.00000E+00] CORRECT: $deriv[$i][$nv][$j][1]\n";
                print IPNODE " The derivative wrt directions 1 & 2 is [ 0.00000E+00] CORRECT: $deriv[$i][$nv][$j][2]\n";
            } 
        }else{
            for my $j (1..3){
                print IPNODE " The number of versions for nj=$j is [1]: $versions[$i]\n";
                for my $nv (1..$versions[$i]) {
                    print IPNODE " For version number $nv: \n";
                    print IPNODE " The Xj($j) coordinate is [ 0.00000E+00] WRONG:   $xyz[$i][$j]\n"; 
                    print IPNODE " The derivative wrt direction 1 is [ 0.00000E+00] WRONG: $deriv[$i][$nv][$j][0]\n";
                    print IPNODE " The derivative wrt direction 2 is [ 0.00000E+00] WRONG: $deriv[$i][$nv][$j][1]\n";
                    print IPNODE " The derivative wrt directions 1 & 2 is [ 0.00000E+00] WRONG: $deriv[$i][$nv][$j][2]\n";
                } 
            } 
        }    
    } 
} 

my $arraysize = @Nodes;
print "Total number of nodes = $arraysize \n";
